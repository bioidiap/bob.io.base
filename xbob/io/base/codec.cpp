/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 14 May 11:53:36 2014 CEST
 *
 * @brief Bindings to bob::io::CodecRegistry
 */

#define XBOB_IO_BASE_MODULE
#include <xbob.io.base/api.h>

int PyBobIoCodec_Register (const char* extension, const char* description, bob::io::file_factory_t factory) {
  boost::shared_ptr<bob::io::CodecRegistry> instance =
    bob::io::CodecRegistry::instance();

  if (instance->isRegistered(extension)) {
    PyErr_Format(PyExc_RuntimeError, "codec for extension `%s' is already registered with description `%s' - in order to register a new codec for such an extension, first unregister the existing codec", extension, PyBobIoCodec_GetDescription(extension));
    return 0;
  }

  instance->registerExtension(extension, description, factory);
  return 1;
}

int PyBobIoCodec_Deregister (const char* extension) {
  boost::shared_ptr<bob::io::CodecRegistry> instance =
    bob::io::CodecRegistry::instance();

  if (!instance->isRegistered(extension)) {
    PyErr_Format(PyExc_RuntimeError, "there is no codec registered for extension `%s'", extension);
    return 0;
  }

  instance->deregisterExtension(extension);
  return 1;
}

int PyBobIoCodec_IsRegistered (const char* extension) {
  boost::shared_ptr<bob::io::CodecRegistry> instance =
    bob::io::CodecRegistry::instance();
  if (instance->isRegistered(extension)) return 1;
  return 0;
}

const char* PyBobIoCodec_GetDescription (const char* extension) {
  boost::shared_ptr<bob::io::CodecRegistry> instance =
    bob::io::CodecRegistry::instance();
  return instance->getDescription(extension);
}
