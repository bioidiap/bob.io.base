.. vim: set fileencoding=utf-8 :
.. Wed 14 May 15:22:33 2014 CEST

.. _bob.io.base:

=========================
 Bob's Core I/O Routines
=========================

This module contains a basic interface to read and write files of various
types.  It provides generic functions :py:func:`bob.io.base.save` and
:py:func:`bob.io.base.load` to write and read various types of data.  In this
interface, data is mainly written using the :py:class:`bob.io.base.HDF5File`
interface.  To enable further types of IO, please import one of the following
packages (the list might not be exhaustive):

* :ref:`bob.io.audio` to load and save audio data of various kinds
* :ref:`bob.io.image` to load and save images of various kinds
* :ref:`bob.io.video` to load and save videos of various types
* :ref:`bob.io.matlab` to load and save matrices in basic matlab ``.mat`` files


Documentation
-------------

.. toctree::
   :maxdepth: 2

   guide
   py_api
   c_cpp_api

TODO
----

.. todolist::

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. include:: links.rst
