/**
 * @author Manuel Gunther
 * @date Tue Sep 13 13:01:31 MDT 2016
 *
 * @brief Tests for bob::io::base
 */

#include <bob.io.base/api.h>
#include <bob.extension/documentation.h>

static auto s_test_api = bob::extension::FunctionDoc(
  "_test_api",
  "Some tests for API functions"
)
.add_prototype("");

static PyObject* _test_api(PyObject*){
BOB_TRY
  Py_RETURN_NONE;
BOB_CATCH_FUNCTION("_test_api", 0)
}

static PyMethodDef module_methods[] = {
    {
      s_test_api.name(),
      (PyCFunction)_test_api,
      METH_NOARGS,
      s_test_api.doc(),
    },
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr, "Tests for bob::io::base");

#if PY_VERSION_HEX >= 0x03000000
static PyModuleDef module_definition = {
  PyModuleDef_HEAD_INIT,
  BOB_EXT_MODULE_NAME,
  module_docstr,
  -1,
  module_methods,
  0, 0, 0, 0
};
#endif

static PyObject* create_module (void) {

# if PY_VERSION_HEX >= 0x03000000
  PyObject* m = PyModule_Create(&module_definition);
  auto m_ = make_xsafe(m);
  const char* ret = "O";
# else
  PyObject* m = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
  const char* ret = "N";
# endif
  if (!m) return 0;

  return Py_BuildValue(ret, m);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
