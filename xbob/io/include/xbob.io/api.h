/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue  5 Nov 12:22:48 2013 
 *
 * @brief C/C++ API for bob::io
 */

#ifndef XBOB_IO_H
#define XBOB_IO_H

#include <xbob.io/config.h>
#include <bob/io/File.h>
#include <boost/preprocessor/stringize.hpp>
#include <boost/shared_ptr.hpp>
#include <Python.h>

#define XBOB_IO_MODULE_PREFIX xbob.io
#define XBOB_IO_MODULE_NAME _library

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

/* Total number of C API pointers */
#define PyXbobIo_API_pointers 2

#ifdef XBOB_IO_MODULE

  /* This section is used when compiling `xbob.core.random' itself */

  /**************
   * Versioning *
   **************/

  extern int PyXbobIo_APIVersion;

  /*****************************
   * Bindings for xbob.io.file *
   *****************************/

  extern PyBobIoFile_Type_TYPE PyBobIoFile_Type;

#else

  /* This section is used in modules that use `blitz.array's' C-API */

/************************************************************************
 * Macros to avoid symbol collision and allow for separate compilation. *
 * We pig-back on symbols already defined for NumPy and apply the same  *
 * set of rules here, creating our own API symbol names.                *
 ************************************************************************/

#  if defined(PY_ARRAY_UNIQUE_SYMBOL)
#    define XBOB_IO_MAKE_API_NAME_INNER(a) XBOB_IO_ ## a
#    define XBOB_IO_MAKE_API_NAME(a) XBOB_IO_MAKE_API_NAME_INNER(a)
#    define PyBlitzArray_API XBOB_IO_MAKE_API_NAME(PY_ARRAY_UNIQUE_SYMBOL)
#  endif

#  if defined(NO_IMPORT_ARRAY)
  extern void **PyXbobIo_API;
#  else
#    if defined(PY_ARRAY_UNIQUE_SYMBOL)
  void **PyXbobIo_API;
#    else
  static void **PyXbobIo_API=NULL;
#    endif
#  endif

  static void **PyXbobIo_API;

  /**************
   * Versioning *
   **************/

# define PyXbobIo_APIVersion (*(PyXbobIo_APIVersion_TYPE *)PyXbobIo_API[PyXbobIo_APIVersion_NUM])

  /*****************************
   * Bindings for xbob.io.file *
   *****************************/

# define PyBobIoFile_Type (*(PyBobIoFile_Type_TYPE *)PyXbobIo_API[PyBobIoFile_Type_NUM])

  /**
   * Returns -1 on error, 0 on success. PyCapsule_Import will set an exception
   * if there's an error.
   */
  static int import_xbob_io(void) {

#if PY_VERSION_HEX >= 0x02070000

    /* New Python API support for library loading */

    PyXbobIo_API = (void **)PyCapsule_Import(BOOST_PP_STRINGIZE(XBOB_IO_MODULE_PREFIX) "." BOOST_PP_STRINGIZE(XBOB_IO_MODULE_NAME) "._C_API", 0);

    if (!PyXbobIo_API) return -1;

#else

    /* Old-style Python API support for library loading */

    PyObject *c_api_object;
    PyObject *module;

    module = PyImport_ImportModule(BOOST_PP_STRINGIZE(XBOB_IO_MODULE_PREFIX) "." BOOST_PP_STRINGIZE(XBOB_IO_MODULE_NAME));

    if (module == NULL) return -1;

    c_api_object = PyObject_GetAttrString(module, "_C_API");

    if (c_api_object == NULL) {
      Py_DECREF(module);
      return -1;
    }

    if (PyCObject_Check(c_api_object)) {
      PyXbobIo_API = (void **)PyCObject_AsVoidPtr(c_api_object);
    }

    Py_DECREF(c_api_object);
    Py_DECREF(module);

#endif
    
    /* Checks that the imported version matches the compiled version */
    int imported_version = *(int*)PyXbobIo_API[PyIo_APIVersion_NUM];

    if (XBOB_IO_API_VERSION != imported_version) {
      PyErr_Format(PyExc_RuntimeError, "%s.%s import error: you compiled against API version 0x%04x, but are now importing an API with version 0x%04x which is not compatible - check your Python runtime environment for errors", BOOST_PP_STRINGIZE(XBOB_IO_MODULE_PREFIX), BOOST_PP_STRINGIZE(XBOB_IO_MODULE_NAME), XBOB_IO_API_VERSION, imported_version);
      return -1;
    }

    /* If you get to this point, all is good */
    return 0;

  }

#endif /* XBOB_IO_MODULE */

#endif /* XBOB_IO_H */
