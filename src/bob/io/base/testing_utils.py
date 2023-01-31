#!/usr/bin/env python
# Andre Anjos <andre.anjos@idiap.ch>
# Thu Feb  7 09:58:22 2013
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Re-usable decorators and utilities for bob test code."""

import functools
import os
import traceback
import unittest

from typing import Optional

import click.testing


def datafile(f, module=None, path="data"):
    """datafile(f, [module], [data]) -> filename.

    Returns the test file on the "data" subdirectory of the current module.

    **Parameters:**

    ``f`` : str
      This is the filename of the file you want to retrieve. Something like ``'movie.avi'``.

    ``module``: str
      [optional] This is the python-style package name of the module you want to retrieve
      the data from. This should be something like ``bob.io.base``, but you
      normally refer it using the ``__name__`` property of the module you want to
      find the path relative to.

    ``path``: str
      [Default: ``'data'``] The subdirectory where the datafile will be taken from inside the module.
      It can be set to ``None`` if it should be taken from the module path root (where the ``__init__.py`` file sits).

    **Returns:**

    ``filename`` : str
      The full path of the file
    """
    resource = __name__ if module is None else module
    final_path = f if path is None else os.path.join(path, f)
    import pkg_resources

    return pkg_resources.resource_filename(resource, final_path)


def temporary_filename(prefix="bobtest_", suffix=".hdf5"):
    """temporary_filename([prefix], [suffix]) -> filename.

    Generates a temporary filename to be used in tests, using the default ``temp`` directory (on Unix-like systems, usually ``/tmp``).
    Please note that you are responsible for deleting the file after your test finished.
    A common way to assure the file to be deleted is:

    .. code-block:: py

       import bob.io.base.testing_utils
       temp = bob.io.base.testing_utils.temporary_filename()
       try:
         # use the temp file
         ...
       finally:
         if os.path.exist(temp): os.remove(temp)

    **Parameters:**

    ``prefix`` : str
      [Default: ``'bobtest_'``] The file name prefix to be added in front of the random file name

    ``suffix`` : str
      [Default: ``'.hdf5'``] The file name extension of the temporary file name

    **Returns:**

    ``filename`` : str
      The name of a temporary file that you can use in your test.
      Don't forget to delete!
    """
    import tempfile

    fd, name = tempfile.mkstemp(suffix, prefix)
    os.close(fd)
    os.unlink(name)
    return name


def extension_available(extension):
    """Decorator to check if a extension is available before enabling a test.

    This decorator is mainly used to decorate a test function, in order to skip tests when the extension is not available.
    The syntax is:

    .. code-block:: py

       import bob.io.base.testing_utils

       @bob.io.base.testing_utils.extension_available('.ext')
       def my_test():
         ...
    """

    def test_wrapper(test):
        @functools.wraps(test)
        def wrapper(*args, **kwargs):
            from . import extensions

            if extension in extensions():
                return test(*args, **kwargs)
            else:
                raise unittest.SkipTest(
                    'Extension to handle "%s" files was not available at compile time'
                    % extension
                )

        return wrapper

    return test_wrapper


def assert_click_runner_result(
    result: click.testing.Result,
    exit_code: int = 0,
    exception_type: Optional[type] = None,
):
    """Helper for asserting click runner results.

    Parameters
    ----------
    result
        The return value on ``click.testing.CLIRunner.invoke()``.
    exit_code
        The expected command exit code (defaults to 0).
    exception_type
        If given, will ensure that the raised exception is of that type.
    """

    m = (
        "Click command exited with code '{}', instead of '{}'.\n"
        "Exception:\n{}\n"
        "Output:\n{}"
    )
    exception = (
        "None"
        if result.exc_info is None
        else "".join(traceback.format_exception(*result.exc_info))
    )
    m = m.format(result.exit_code, exit_code, exception, result.output)
    assert result.exit_code == exit_code, m
    if exit_code == 0:
        assert not result.exception, m
    if exception_type is not None:
        assert isinstance(result.exception, exception_type), m
