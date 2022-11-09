import numpy as np

from PIL import Image


def to_matplotlib(img):
    """Returns a view of the image from Bob format to matplotlib format. This
    function works with images, batches of images, videos, and higher
    dimensional arrays that contain images.

    Parameters
    ----------
    img : numpy.ndarray
        A N dimensional array containing an image in Bob format (channels
        first): For an ND array (N >= 3), the image should have the following
        format: ``(..., c, h, w)``.

    Returns
    -------
    numpy.ndarray
        A view of the ``img`` compatible with
        :py:func:`matplotlib.pyplot.imshow`.
    """
    if img.ndim < 3:
        return img
    return np.moveaxis(img, -3, -1)


def to_bob(img):
    """Returns a view of the image from matplotlib format to Bob format. This
    function works with images, batches of images, videos, and higher
    dimensional arrays that contain images.

    Parameters
    ----------
    img : numpy.ndarray
        An image in matplotlib format (channels last): For an ND array (N >= 3),
        the image should have the following format: ``(..., h, w, c)``.

    Returns
    -------
    numpy.ndarray
        A view of the ``img`` compatible with Bob.
    """
    if img.ndim < 3:
        return img
    return np.moveaxis(img, -1, -3)


def bob_to_pillow(img):
    """Converts the floating or uint8 image to a Pillow Image.

    Parameters
    ----------
    img : numpy.ndarray
        A gray-scale or RGB color image in Bob format (channels first)

    Returns
    -------
    PIL.Image.Image
        An Image object of pillow.
    """
    # first convert to matplotlib format
    img = to_matplotlib(img)
    # if img is floating point, convert to uint8
    if isinstance(img, np.floating):
        # In Bob, we expect float images to be between 0 and 255
        # see https://gitlab.idiap.ch/bob/bob.bio.face/-/issues/48
        img = np.round(img)
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)

    return Image.fromarray(img)


def pillow_to_bob(img):
    """Converts an RGB or gray-scale pillow image to Bob format.

    Parameters
    ----------
    img: PIL.Image.Image
        A Pillow Image

    Returns
    -------
    numpy.ndarray
        Image in Bob format
    """
    return to_bob(np.array(img))


def opencvbgr_to_bob(img):
    """Returns a view of the image from OpenCV BGR format to Bob RGB format.
    This function works with images, batches of images, videos, and higher
    dimensional arrays that contain images.

    Parameters
    ----------
    img : numpy.ndarray
        An image loaded by OpenCV. It needs to have at least 3 dimensions.

    Returns
    -------
    numpy.ndarray
        A view of the ``img`` compatible with Bob.

    Raises
    ------
    ValueError
        If the image dimension is less than 3.
    """
    if img.ndim < 3:
        return img
    img = img[..., ::-1]
    return to_bob(img)


def bob_to_opencvbgr(img):
    """Returns a view of the image from Bob format to OpenCV BGR format. This
    function works with images, batches of images, videos, and higher
    dimensional arrays that contain images.

    Parameters
    ----------
    img : numpy.ndarray
        An image loaded by Bob. It needs to have at least 3 dimensions.

    Returns
    -------
    numpy.ndarray
        A view of the ``img`` compatible with OpenCV.

    Raises
    ------
    ValueError
        If the image dimension is less than 3.
    """
    if img.ndim < 3:
        return img
    img = img[..., ::-1, :, :]
    return to_matplotlib(img)


def imshow(img, cmap=None, **kwargs):
    """Plots the images that are returned by :py:func:`bob.io.base.load`

    Parameters
    ----------
    img : numpy.ndarray
        A 2 or 3 dimensional array containing an image in
        bob style: For a 2D array (grayscale image) should be ``(h, w)``;
        A 3D array (color image) should be in the ``(c, h, w)`` format.
    cmap : matplotlib.colors.Colormap
        Colormap, optional, default: ``None``.
        If ``cmap`` is ``None`` and ``img.ndim`` is 2, defaults to 'gray'.
        ``cmap`` is ignored when ``img`` has RGB(A) information.
    **kwargs
        These are passed directly to :py:func:`matplotlib.pyplot.imshow`

    Returns
    -------
    object
        Returns whatever ``plt.imshow`` returns.
    """
    import matplotlib.pyplot as plt

    if cmap is None and img.ndim == 2:
        cmap = "gray"

    return plt.imshow(to_matplotlib(img), cmap=cmap, **kwargs)
