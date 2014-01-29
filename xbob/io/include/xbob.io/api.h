/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue  5 Nov 12:22:48 2013
 *
 * @brief C/C++ API for bob::io
 */

#ifndef XBOB_IO_H
#define XBOB_IO_H

/* Define Module Name and Prefix for other Modules
   Note: We cannot use XBOB_EXT_* macros here, unfortunately */
#define XBOB_IO_PREFIX    "xbob.io"
#define XBOB_IO_FULL_NAME "xbob.io._library"

#include <Python.h>

#include <xbob.io/config.h>
#include <bob/config.h>
#include <bob/io/File.h>
#include <bob/io/HDF5File.h>

#if WITH_FFMPEG
#include <bob/io/VideoReader.h>
#include <bob/io/VideoWriter.h>
#endif /* WITH_FFMPEG */

#include <boost/shared_ptr.hpp>

/*******************
 * C API functions *
 *******************/

/**************
 * Versioning *
 **************/

#define PyXbobIo_APIVersion_NUM 0
#define PyXbobIo_APIVersion_TYPE int

/*****************************
 * Bindings for xbob.io.file *
 *****************************/

/* Type definition for PyBobIoFileObject */
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  boost::shared_ptr<bob::io::File> f;

} PyBobIoFileObject;

#define PyBobIoFile_Type_NUM 1
#define PyBobIoFile_Type_TYPE PyTypeObject

typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  PyBobIoFileObject* pyfile;
  Py_ssize_t curpos;

} PyBobIoFileIteratorObject;

#define PyBobIoFileIterator_Type_NUM 2
#define PyBobIoFileIterator_Type_TYPE PyTypeObject

/************************
 * I/O generic bindings *
 ************************/

#define PyBobIo_AsTypenum_NUM 3
#define PyBobIo_AsTypenum_RET int
#define PyBobIo_AsTypenum_PROTO (bob::core::array::ElementType et)

#define PyBobIo_TypeInfoAsTuple_NUM 4
#define PyBobIo_TypeInfoAsTuple_RET PyObject*
#define PyBobIo_TypeInfoAsTuple_PROTO (const bob::core::array::typeinfo& ti)

#define PyBobIo_FilenameConverter_NUM 5
#define PyBobIo_FilenameConverter_RET int
#define PyBobIo_FilenameConverter_PROTO (PyObject* o, PyObject** b)

/*****************
 * HDF5 bindings *
 *****************/

typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  boost::shared_ptr<bob::io::HDF5File> f;

} PyBobIoHDF5FileObject;

#define PyBobIoHDF5File_Type_NUM 6
#define PyBobIoHDF5File_Type_TYPE PyTypeObject

#define PyBobIoHDF5File_Check_NUM 7
#define PyBobIoHDF5File_Check_RET int
#define PyBobIoHDF5File_Check_PROTO (PyObject* o)

#define PyBobIoHDF5File_Converter_NUM 8
#define PyBobIoHDF5File_Converter_RET int
#define PyBobIoHDF5File_Converter_PROTO (PyObject* o, PyBobIoHDF5FileObject** a)

#if WITH_FFMPEG

/******************
 * Video bindings *
 ******************/

typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  boost::shared_ptr<bob::io::VideoReader> v;

} PyBobIoVideoReaderObject;

#define PyBobIoVideoReader_Type_NUM 9
#define PyBobIoVideoReader_Type_TYPE PyTypeObject

typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  PyBobIoVideoReaderObject* pyreader;
  boost::shared_ptr<bob::io::VideoReader::const_iterator> iter;

} PyBobIoVideoReaderIteratorObject;

#define PyBobIoVideoReaderIterator_Type_NUM 10
#define PyBobIoVideoReaderIterator_Type_TYPE PyTypeObject

typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  boost::shared_ptr<bob::io::VideoWriter> v;

} PyBobIoVideoWriterObject;

#define PyBobIoVideoWriter_Type_NUM 11
#define PyBobIoVideoWriter_Type_TYPE PyTypeObject

#endif /* WITH_FFMPEG */

/* Total number of C API pointers */
#if WITH_FFMPEG
#  define PyXbobIo_API_pointers 12
#else
#  define PyXbobIo_API_pointers 9
#endif /* WITH_FFMPEG */

#ifdef XBOB_IO_MODULE

  /* This section is used when compiling `xbob.io' itself */

  /**************
   * Versioning *
   **************/

  extern int PyXbobIo_APIVersion;

  /*****************************
   * Bindings for xbob.io.file *
   *****************************/

  extern PyBobIoFile_Type_TYPE PyBobIoFile_Type;
  extern PyBobIoFileIterator_Type_TYPE PyBobIoFileIterator_Type;

  /************************
   * I/O generic bindings *
   ************************/

  PyBobIo_AsTypenum_RET PyBobIo_AsTypenum PyBobIo_AsTypenum_PROTO;

  PyBobIo_TypeInfoAsTuple_RET PyBobIo_TypeInfoAsTuple PyBobIo_TypeInfoAsTuple_PROTO;

  PyBobIo_FilenameConverter_RET PyBobIo_FilenameConverter PyBobIo_FilenameConverter_PROTO;

/*****************
 * HDF5 bindings *
 *****************/

  extern PyBobIoHDF5File_Type_TYPE PyBobIoHDF5File_Type;

  PyBobIoHDF5File_Check_RET PyBobIoHDF5File_Check PyBobIoHDF5File_Check_PROTO;

  PyBobIoHDF5File_Converter_RET PyBobIoHDF5File_Converter PyBobIoHDF5File_Converter_PROTO;

#if WITH_FFMPEG
  /******************
   * Video bindings *
   ******************/

  extern PyBobIoVideoReader_Type_TYPE PyBobIoVideoReader_Type;

  extern PyBobIoVideoReaderIterator_Type_TYPE PyBobIoVideoReaderIterator_Type;

  extern PyBobIoVideoWriter_Type_TYPE PyBobIoVideoWriter_Type;
#endif /* WITH_FFMPEG */

#else

  /* This section is used in modules that use `xbob.io's' C-API */

#  if defined(NO_IMPORT_ARRAY)
  extern void **PyXbobIo_API;
#  else
#    if defined(PY_ARRAY_UNIQUE_SYMBOL)
  void **PyXbobIo_API;
#    else
  static void **PyXbobIo_API=NULL;
#    endif
#  endif

  /**************
   * Versioning *
   **************/

# define PyXbobIo_APIVersion (*(PyXbobIo_APIVersion_TYPE *)PyXbobIo_API[PyXbobIo_APIVersion_NUM])

  /*****************************
   * Bindings for xbob.io.file *
   *****************************/

# define PyBobIoFile_Type (*(PyBobIoFile_Type_TYPE *)PyXbobIo_API[PyBobIoFile_Type_NUM])
# define PyBobIoFileIterator_Type (*(PyBobIoFileIterator_Type_TYPE *)PyXbobIo_API[PyBobIoFileIterator_Type_NUM])

  /************************
   * I/O generic bindings *
   ************************/

# define PyBobIo_AsTypenum (*(PyBobIo_AsTypenum_RET (*)PyBobIo_AsTypenum_PROTO) PyXbobIo_API[PyBobIo_AsTypenum_NUM])

# define PyBobIo_TypeInfoAsTuple (*(PyBobIo_TypeInfoAsTuple_RET (*)PyBobIo_TypeInfoAsTuple_PROTO) PyXbobIo_API[PyBobIo_TypeInfoAsTuple_NUM])

# define PyBobIo_FilenameConverter (*(PyBobIo_FilenameConverter_RET (*)PyBobIo_FilenameConverter_PROTO) PyXbobIo_API[PyBobIo_FilenameConverter_NUM])

  /*****************
   * HDF5 bindings *
   *****************/

# define PyBobIoHDF5File_Type (*(PyBobIoHDF5File_Type_TYPE *)PyXbobIo_API[PyBobIoHDF5File_Type_NUM])

# define PyBobIoHDF5File_Check (*(PyBobIoHDF5File_Check_RET (*)PyBobIoHDF5File_Check_PROTO) PyXbobIo_API[PyBobIoHDF5File_Check_NUM])

# define PyBobIoHDF5File_Converter (*(PyBobIoHDF5File_Converter_RET (*)PyBobIoHDF5File_Converter_PROTO) PyXbobIo_API[PyBobIoHDF5File_Converter_NUM])

#if WITH_FFMPEG
  /******************
   * Video bindings *
   ******************/

# define PyBobIoVideoReader_Type (*(PyBobIoVideoReader_Type_TYPE *)PyXbobIo_API[PyBobIoVideoReader_Type_NUM])

# define PyBobIoVideoReaderIterator_Type (*(PyBobIoVideoReaderIterator_Type_TYPE *)PyXbobIo_API[PyBobIoVideoReaderIterator_Type_NUM])

# define PyBobIoVideoWriterIterator_Type (*(PyBobIoVideoWriterIterator_Type_TYPE *)PyXbobIo_API[PyBobIoVideoWriterIterator_Type_NUM])
#endif /* WITH_FFMPEG */

# if !defined(NO_IMPORT_ARRAY)

#include <xbob.blitz/capi.h>

  /**
   * Returns -1 on error, 0 on success.
   */
  static int import_xbob_io(void) {

    PyObject *c_api_object;
    PyObject *module;

    module = PyImport_ImportModule(XBOB_IO_FULL_NAME);

    if (module == NULL) return -1;

    c_api_object = PyObject_GetAttrString(module, "_C_API");

    if (c_api_object == NULL) {
      Py_DECREF(module);
      return -1;
    }

#   if PY_VERSION_HEX >= 0x02070000
    if (PyCapsule_CheckExact(c_api_object)) {
      PyXbobIo_API = (void **)PyCapsule_GetPointer(c_api_object,
          PyCapsule_GetName(c_api_object));
    }
#   else
    if (PyCObject_Check(c_api_object)) {
      PyXbobIo_API = (void **)PyCObject_AsVoidPtr(c_api_object);
    }
#   endif

    Py_DECREF(c_api_object);
    Py_DECREF(module);

    if (!PyXbobIo_API) {
      PyErr_SetString(PyExc_ImportError, "cannot find C/C++ API "
#   if PY_VERSION_HEX >= 0x02070000
          "capsule"
#   else
          "cobject"
#   endif
          " at `" XBOB_IO_FULL_NAME "._C_API'");
      return -1;
    }

    /* Checks that the imported version matches the compiled version */
    int imported_version = *(int*)PyXbobIo_API[PyXbobIo_APIVersion_NUM];

    if (XBOB_IO_API_VERSION != imported_version) {
      PyErr_Format(PyExc_ImportError, XBOB_IO_FULL_NAME " import error: you compiled against API version 0x%04x, but are now importing an API with version 0x%04x which is not compatible - check your Python runtime environment for errors", XBOB_IO_API_VERSION, imported_version);
      return -1;
    }

    /* Imports the xbob.blitz C-API */
    if (import_xbob_blitz() < 0) {
      PyErr_Print();
      PyErr_SetString(PyExc_ImportError, "xbob.blitz failed to import");
      return -1;
    }

    /* If you get to this point, all is good */
    return 0;

  }

# endif //!defined(NO_IMPORT_ARRAY)

#endif /* XBOB_IO_MODULE */

#endif /* XBOB_IO_H */
