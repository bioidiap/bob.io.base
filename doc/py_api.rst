.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Sat 16 Nov 20:52:58 2013

============
 Python API
============

This section includes information for using the pure Python API of ``bob.io.base``.


Classes
-------

.. autosummary::
   bob.io.base.File
   bob.io.base.HDF5File

Functions
---------

.. autosummary::
   bob.io.base.load
   bob.io.base.merge
   bob.io.base.save
   bob.io.base.append
   bob.io.base.peek
   bob.io.base.peek_all
   bob.io.base.create_directories_safe
   bob.io.base.vstack_features
   bob.io.base.from_dict_to_hdf5
   bob.io.base.from_hdf5_to_dict      

   bob.io.base.extensions
   bob.io.base.get_config

Test Utilities
--------------

These functions might be useful when you are writing your nose tests.
Please note that this is not part of the default ``bob.io.base`` API, so in order to use it, you have to ``import bob.io.base.test_utils`` separately.

.. autosummary::
   bob.io.base.test_utils.datafile
   bob.io.base.test_utils.temporary_filename
   bob.io.base.test_utils.extension_available


Details
-------

.. automodule:: bob.io.base
   :inherited-members:

.. automodule:: bob.io.base.test_utils
