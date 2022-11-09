#!/usr/bin/env python
# Laurent El Shafey <laurent.el-shafey@idiap.ch>
# Wed Aug 14 12:27:57 CEST 2013
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Runs some image tests."""

import os

import numpy

# import bob.io.image
import pytest

from bob.io.base import load, write
from bob.io.base.testing_utils import datafile, temporary_filename

# These are some global parameters for the test.
PNG_INDEXED_COLOR = datafile("img_indexed_color.png", __name__)
PNG_INDEXED_COLOR_ALPHA = datafile("img_indexed_color_alpha.png", __name__)
PNG_RGBA_COLOR = datafile("img_rgba_color.png", __name__)
PNG_GRAY_ALPHA = datafile("img_gray_alpha.png", __name__)
PNG_tRNS = datafile("img_trns.png", __name__)


def test_png_indexed_color():
    # Read an indexed color PNG image, and compared with hardcoded values

    img = load(PNG_INDEXED_COLOR)
    assert img.shape == (3, 22, 32)
    assert img[0, 0, 0] == 255
    assert img[0, 17, 17] == 117


def test_png_rgba_color():
    # Read an indexed color PNG image, and compared with hardcoded values
    img = load(PNG_RGBA_COLOR)
    assert img.shape == (3, 22, 32)
    assert img[0, 0, 0] == 255
    assert img[0, 17, 17] == 117


def test_png_indexed_color_alpha():
    # Read an indexed color+alpha PNG image, and compared with hardcoded values
    img = load(PNG_INDEXED_COLOR_ALPHA)
    assert img.shape == (3, 22, 32)
    assert img[0, 0, 0] == 255
    assert img[0, 17, 17] == 117


def test_png_indexed_trns():
    # Read an tRNS PNG image (without alpha), and compared with hardcoded values
    img = load(PNG_tRNS)
    assert img.shape == (3, 22, 32)
    assert img[0, 0, 0] == 255
    assert img[0, 17, 17] == 117


def test_png_gray_alpha():
    # Read a gray+alpha PNG image, and compared with hardcoded values

    img = load(PNG_GRAY_ALPHA)

    assert img.shape == (22, 32)
    assert img[0, 0] == 255
    assert img[17, 17] == 51


def transcode(filename):
    tmpname = temporary_filename(suffix=os.path.splitext(filename)[1])
    tmpnam_ = temporary_filename(suffix=os.path.splitext(filename)[1])

    try:
        # complete transcoding test
        image = load(filename)

        # save with the same extension
        write(image, tmpname)

        # reload the image from the file
        image2 = load(tmpname)

        assert numpy.array_equal(image, image2)

        # test getting part of the image as well
        if len(image.shape) == 3:
            subsample = image[:, ::2, ::2]
        else:
            subsample = image[::2, ::2]

        assert not subsample.flags.contiguous
        write(subsample, tmpnam_)
        image3 = load(tmpnam_)
        assert numpy.array_equal(subsample, image3)

    finally:
        if os.path.exists(tmpname):
            os.unlink(tmpname)
        if os.path.exists(tmpnam_):
            os.unlink(tmpnam_)


def test_netpbm():
    transcode(datafile("test.pbm", __name__))  # indexed, works fine
    transcode(datafile("test.pgm", __name__))  # indexed, works fine
    transcode(datafile("test.ppm", __name__))  # indexed, works fine
    transcode(datafile("test_2.pgm", __name__))  # indexed, works fine
    # transcode(datafile("test_2.ppm", __name__))  # indexed, works fine ----> THIS DOES NOT WORK
    transcode(datafile("test_spaces.pgm", __name__))  # indexed, works fine

    # transcode(datafile("test.jpg", __name__))  # does not work
    # because of re-compression


def notest_gif():
    transcode(datafile("test.gif", __name__))


def test_image_load():
    # test that the generic bob.io.image.load function works as expected
    for filename in (
        "test.jpg",
        "cmyk.jpg",
        "test.pbm",
        "test_corrupted.pbm",
        "test.pgm",
        "test_corrupted.pgm",
        "test.ppm",
        "test_corrupted.ppm",
        "test.gif",
    ):

        full_file = datafile(filename, __name__)

        # load with just image name
        i1 = load(full_file)

        assert i1.shape == (6, 4)

    # Loading the last pgm file
    full_file = datafile("test_spaces.pgm", __name__)
    load(full_file).shape == (100, 100)

    # Testing exception
    with pytest.raises(RuntimeError):
        load(os.path.splitext(full_file)[0] + ".unknown")


def test_image_exceptions():
    # This tests some image exceptions I found

    # Real GRAY PNG image
    transcode(datafile("read_png_gray.png", __name__))

    # Trucated JPEG
    # THIS TEST FAILS
    # transcode(datafile("truncated_jpeg.jpg", __name__))
