#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Tests for the base HDF5 infrastructure
"""

import os
from bob.io.base import load, save
from ..test_utils import temporary_filename


import numpy as np
import random


def read_write_check(data):
    """Testing loading and save different file types"""

    tmpname = temporary_filename()

    try:

        save(data, tmpname)
        data2 = load(tmpname)
    finally:
        os.unlink(tmpname)

    assert np.allclose(data, data2, atol=10e-5, rtol=10e-5)


def test_type_support():

    # This test will go through all supported types for reading/writing data
    # from to HDF5 files. One single file will hold all data for this test.
    # This is also supported with HDF5: multiple variables in a single file.

    N = 100

    data = [int(random.uniform(0, 100)) for z in range(N)]

    read_write_check(np.array(data, np.int8))
    read_write_check(np.array(data, np.uint8))
    read_write_check(np.array(data, np.int16))
    read_write_check(np.array(data, np.uint16))
    read_write_check(np.array(data, np.int32))
    read_write_check(np.array(data, np.uint32))
    read_write_check(np.array(data, np.int64))
    read_write_check(np.array(data, np.uint64))
    read_write_check(np.array(data, np.float32))
    read_write_check(np.array(data, np.float64))
    read_write_check(np.array(data, np.complex64))
    read_write_check(np.array(data, np.complex128))
