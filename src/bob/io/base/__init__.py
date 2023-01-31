# import Libraries of other lib packages
import logging

import h5py
import imageio
import numpy as np

logger = logging.getLogger(__name__)
import os

# Allowing the loading of truncated files in case PIL is used
# https://github.com/kirumang/Pix2Pose/issues/2
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


hdf5_extensions = [".hdf5", ".h5", ".hdf", ".hdf5", ".h5", ".hdf", ".hdf5"]
image_extensions = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".pgm",
    ".pbm",
    ".pnm",
    ".ppm",
]


def _is_string(s):
    """Returns ``True`` if the given object is a string or bytes."""
    return isinstance(s, (bytes, str))


@np.deprecate(new_name="os.makedirs(directory, exist_ok=True)")
def create_directories_safe(directory, dryrun=False):
    """Creates a directory if it does not exists, with concurrent access
    support. This function will also create any parent directories that might
    be required. If the dryrun option is selected, it does not actually create
    the directory, but just writes the (Linux) command that would have been
    executed.

    **Parameters:**

    ``directory`` : str
      The directory that you want to create.

    ``dryrun`` : bool
      Only ``print`` the command to console, but do not execute it.
    """
    if dryrun:
        print("[dry-run] mkdir -p '%s'" % directory)
    else:
        os.makedirs(directory, exist_ok=True)


def open_file(filename) -> np.ndarray:
    """Reads a file content.

    Parameters
    ----------

    ``filename`` : str
      The name of the file to open.
    """

    def check_gray(img):
        # Checking for gray scaled images
        if (
            img.ndim > 2
            and np.array_equal(img[:, :, 0], img[:, :, 1])
            and np.array_equal(img[:, :, 0], img[:, :, 2])
        ):
            img = img[:, :, 0]
        return img

    # get the extension
    extension = os.path.splitext(filename)[1].lower()

    if extension in hdf5_extensions:
        with h5py.File(filename, "r") as f:
            keys = list(f.keys())
            if len(keys) == 1:
                key = keys[0]
            else:
                key = "array"
            if key not in keys:
                raise RuntimeError(
                    f"The file {filename} does not contain the key {key}"
                )
            dataset = f[key]
            # if the data was saved as a string, load it back as string
            string_dtype = h5py.check_string_dtype(dataset.dtype)
            if string_dtype is not None:
                dataset = dataset.asstr()
            return dataset[()]

    elif extension in image_extensions:
        from ..image import to_bob

        img = imageio.imread(filename)

        # PNGs have a 4th channel, which we don't want
        # Alpha channels for instance have to be ignored
        if img.ndim > 2:
            if extension.lower() == ".png":
                img = img[:, :, 0:3]

        img = check_gray(img)
        return img if img.ndim == 2 else to_bob(img)
    else:
        raise ValueError(f"Unknown file extension: {extension}")


def write_file(filename, data, format="pillow") -> None:
    """Writes the contents of a :py:class:`numpy.ndarray` to a file.

    Parameters
    ----------

    ``filename`` : str
      The name of the file to write to.

    ``data`` : :py:class:`numpy.ndarray`
      The data to write to the file.

    ``format`` : str
      The format to use to read the file. By default imageio selects the appropriate for you based on the filename and its contents
    """
    extension = os.path.splitext(filename)[1]  # get the extension

    if extension in hdf5_extensions:
        with h5py.File(filename, "w") as f:
            f["array"] = data
    elif extension in image_extensions:
        # Pillow is the format with the best support for all image formats
        from ..image import to_matplotlib

        imageio.imwrite(filename, to_matplotlib(data), format=format)
    else:
        raise RuntimeError(f"Unknown file extension: {extension}")


def load(inputs) -> np.ndarray:
    """Loads the content of a file.

    Will take a filename (or an iterable of filenames) and put the content into a
    :py:class:`numpy.ndarray`.

    **Parameters:**

    ``inputs`` : various types

      This might represent several different entities:

      1. The name of a file (full path) from where to load the data. In this
         case, this assumes that the file contains an array and returns a loaded
         numpy ndarray.
      2. An iterable of filenames to be loaded in memory. In this case, this
         would assume that each file contains a single 1D sample or a set of 1D
         samples, load them in memory and concatenate them into a single and
         returned 2D :py:class:`numpy.ndarray`.

    **Returns:**

    ``data`` : :py:class:`numpy.ndarray`
      The data loaded from the given ``inputs``.
    """
    from collections.abc import Iterable

    import numpy

    if _is_string(inputs):
        if not os.path.exists(inputs):
            raise RuntimeError(f"`{inputs}' does not exist!")
        try:
            return open_file(inputs)
        except Exception as e:
            raise RuntimeError(f"Could not load `{inputs}'!") from e

    elif isinstance(inputs, Iterable):
        retval = []
        for obj in inputs:
            if _is_string(obj):
                retval.append(load(obj))
            else:
                raise TypeError(
                    "Iterable contains an object which is not a filename"
                )
        return numpy.vstack(retval)
    else:
        raise TypeError(
            "Unexpected input object. This function is expecting a filename, "
            "or an iterable of filenames."
        )


def save(array, filename, create_directories=False):
    """Saves the contents of an array-like object to file.

    Effectively, this is the same as opening a file with the mode flag set to ``'w'``
    (write with truncation) and calling ``file.write`` passing ``array`` as parameter.

    Parameters:

    ``array`` : array_like
      The array-like object to be saved on the file

    ``filename`` : str
      The name of the file where you need the contents saved to

    ``create_directories`` : bool
      Automatically generate the directories if required (defaults to ``False``
      because of compatibility reasons; might change in future to default to
      ``True``)
    """
    # create directory if not existent yet
    if create_directories:
        create_directories_safe(os.path.dirname(filename))

    # if array is a string, don't create a numpy array
    if not isinstance(array, str):
        # requires data is c-contiguous and aligned, will create a copy otherwise
        array = np.require(array, requirements=("C_CONTIGUOUS", "ALIGNED"))

    write_file(filename, array)


# Just to make it homogenous with the C++ API
write = save
read = load


# Keeps compatibility with the previously existing API
# open = File


def _generate_features(reader, paths, same_size=False):
    """Load and stack features in a memory efficient way. This function is
    meant to be used inside :py:func:`vstack_features`.

    Parameters
    ----------
    reader : ``collections.Callable``
      See the documentation of :py:func:`vstack_features`.
    paths : ``collections.Iterable``
      See the documentation of :py:func:`vstack_features`.
    same_size : :obj:`bool`, optional
      See the documentation of :py:func:`vstack_features`.

    Yields
    ------
    object
      The first object returned is a tuple of :py:class:`numpy.dtype` of
      features and the shape of the first feature. The rest of objects are
      the actual values in features. The features are returned in C order.
    """
    shape_determined = False
    for i, path in enumerate(paths):

        feature = np.atleast_2d(reader(path))
        feature = np.ascontiguousarray(feature)
        if not shape_determined:
            shape_determined = True
            dtype = feature.dtype
            shape = list(feature.shape)
            yield (dtype, shape)
        else:
            # make sure all features have the same shape and dtype
            if same_size:
                assert shape == list(
                    feature.shape
                ), f"Expected feature shape of {shape}, got {feature.shape}"
            else:
                assert shape[1:] == list(
                    feature.shape[1:]
                ), f"Ignoring first dimension, expected feature shape of {shape}, got {feature.shape}"
            assert dtype == feature.dtype

        if same_size:
            yield (feature.ravel(),)
        else:
            for feat in feature:
                yield (feat.ravel(),)


def vstack_features(reader, paths, same_size=False, dtype=None):
    """Stacks all features in a memory efficient way.

    Parameters
    ----------
    reader : ``collections.Callable``
      The function to load the features. The function should only take one
      argument ``path`` and return loaded features. Use :any:`functools.partial`
      to accommodate your reader to this format.
      The features returned by ``reader`` are expected to have the same
      :py:class:`numpy.dtype` and the same shape except for their first
      dimension. First dimension should correspond to the number of samples.
    paths : ``collections.Iterable``
      An iterable of paths to iterate on. Whatever is inside path is given to
      ``reader`` so they do not need to be necessarily paths to actual files.
      If ``same_size`` is ``True``, ``len(paths)`` must be valid.
    same_size : :obj:`bool`, optional
      If ``True``, it assumes that arrays inside all the paths are the same
      shape. If you know the features are the same size in all paths, set this
      to ``True`` to improve the performance.
    dtype : :py:class:`numpy.dtype`, optional
      If provided, the data will be casted to this format.

    Returns
    -------
    numpy.ndarray
      The read features with the shape ``(n_samples, *features_shape[1:])``.

    Examples
    --------
    This function in a simple way is equivalent to calling
    ``numpy.vstack([reader(p) for p in paths])``.

    >>> import numpy
    >>> from bob.io.base import vstack_features
    >>> def reader(path):
    ...     # in each file, there are 5 samples and features are 2 dimensional.
    ...     return numpy.arange(10).reshape(5,2)
    >>> paths = ['path1', 'path2']
    >>> all_features = vstack_features(reader, paths)
    >>> numpy.allclose(all_features, numpy.array(
    ...     [[0, 1],
    ...      [2, 3],
    ...      [4, 5],
    ...      [6, 7],
    ...      [8, 9],
    ...      [0, 1],
    ...      [2, 3],
    ...      [4, 5],
    ...      [6, 7],
    ...      [8, 9]]))
    True
    >>> all_features_with_more_memory = numpy.vstack([reader(p) for p in paths])
    >>> numpy.allclose(all_features, all_features_with_more_memory)
    True

    You can allocate the array at once to improve the performance if you know
    that all features in paths have the same shape and you know the total number
    of the paths:

    >>> all_features = vstack_features(reader, paths, same_size=True)
    >>> numpy.allclose(all_features, numpy.array(
    ...     [[0, 1],
    ...      [2, 3],
    ...      [4, 5],
    ...      [6, 7],
    ...      [8, 9],
    ...      [0, 1],
    ...      [2, 3],
    ...      [4, 5],
    ...      [6, 7],
    ...      [8, 9]]))
    True
    """
    iterable = _generate_features(reader, paths, same_size)
    data_dtype, shape = next(iterable)
    if dtype is None:
        dtype = data_dtype
    if same_size:
        # numpy black magic: https://stackoverflow.com/a/12473478/1286165
        field_dtype = [("", (dtype, (np.prod(shape),)))]
        total_size = len(paths)
        all_features = np.fromiter(iterable, field_dtype, total_size)
    else:
        field_dtype = [("", (dtype, (np.prod(shape[1:]),)))]
        all_features = np.fromiter(iterable, field_dtype)

    # go from a field array to a normal array
    all_features = all_features.view(dtype)
    # the shape is assumed to be (n_samples, ...) it can be (5, 2) or (5, 3, 4).
    shape = list(shape)
    shape[0] = -1
    return np.reshape(all_features, shape, order="C")


# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith("_")]
