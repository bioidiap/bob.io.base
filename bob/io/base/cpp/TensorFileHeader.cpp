/**
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * This class defines an header for storing multiarrays into .tensor files.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#define BOB_IO_BASE_MODULE
#include "TensorFileHeader.h"
#include <bob.blitz/capi.h>
#include <boost/format.hpp>

bob::io::base::detail::TensorFileHeader::TensorFileHeader()
  : m_tensor_type(bob::io::base::Char),
    m_type(),
    m_n_samples(0),
    m_tensor_size(0)
{
}

bob::io::base::detail::TensorFileHeader::~TensorFileHeader() { }

size_t bob::io::base::detail::TensorFileHeader::getArrayIndex (size_t index) const {
  size_t header_size = 7 * sizeof(int);
  return header_size + index * m_tensor_size;
}

void bob::io::base::detail::TensorFileHeader::read(std::istream& str) {
  // Start reading at the beginning of the stream
  str.seekg(std::ios_base::beg);

  int val;
  str.read(reinterpret_cast<char*>(&val), sizeof(int));
  m_tensor_type = (bob::io::base::TensorType)val;
  m_type.dtype = bob::io::base::tensorTypeToArrayType(m_tensor_type);

  str.read(reinterpret_cast<char*>(&val), sizeof(int));
  m_n_samples = (size_t)val;

  str.read(reinterpret_cast<char*>(&val), sizeof(int));
  size_t nd = (size_t)val;

  size_t shape[BOB_BLITZ_MAXDIMS];

  str.read(reinterpret_cast<char*>(&val), sizeof(int));
  shape[0] = (size_t)val;
  str.read(reinterpret_cast<char*>(&val), sizeof(int));
  shape[1] = (size_t)val;
  str.read( reinterpret_cast<char*>(&val), sizeof(int));
  shape[2] = (size_t)val;
  str.read( reinterpret_cast<char*>(&val), sizeof(int));
  shape[3] = (size_t)val;

  BobIoTypeinfo_Set(&m_type, m_type.dtype, nd, shape);

  header_ok();
}

void bob::io::base::detail::TensorFileHeader::write(std::ostream& str) const
{
  // Start writing at the beginning of the stream
  str.seekp(std::ios_base::beg);

  int val;
  val = (int)m_tensor_type;
  str.write(reinterpret_cast<char*>(&val), sizeof(int));
  val = (int)m_n_samples;
  str.write(reinterpret_cast<char*>(&val), sizeof(int));
  val = (int)m_type.nd;
  str.write(reinterpret_cast<char*>(&val), sizeof(int));
  val = (int)m_type.shape[0];
  str.write(reinterpret_cast<char*>(&val), sizeof(int));
  val = (int)m_type.shape[1];
  str.write(reinterpret_cast<char*>(&val), sizeof(int));
  val = (int)m_type.shape[2];
  str.write(reinterpret_cast<char*>(&val), sizeof(int));
  val = (int)m_type.shape[3];
  str.write(reinterpret_cast<char*>(&val), sizeof(int));
}

void bob::io::base::detail::TensorFileHeader::header_ok()
{
  // Check the type
  switch (m_tensor_type)
  {
    // supported tensor types
    case bob::io::base::Char:
    case bob::io::base::Short:
    case bob::io::base::Int:
    case bob::io::base::Long:
    case bob::io::base::Float:
    case bob::io::base::Double:
      break;
    // error
    default:
      throw std::runtime_error("unsupported data type found while scanning header of tensor file");
  }

  // Check the number of samples and dimensions
  if( m_type.nd < 1 || m_type.nd > 4) {
    boost::format m("header for tensor file indicates an unsupported type: %s");
    m % BobIoTypeinfo_Str(&m_type);
    throw std::runtime_error(m.str());
  }

  // OK
  update();
}

void bob::io::base::detail::TensorFileHeader::update()
{
  size_t base_size = 0;
  switch (m_tensor_type)
  {
    case bob::io::base::Char:    base_size = sizeof(char); break;
    case bob::io::base::Short:   base_size = sizeof(short); break;
    case bob::io::base::Int:     base_size = sizeof(int); break;
    case bob::io::base::Long:    base_size = sizeof(long); break;
    case bob::io::base::Float:   base_size = sizeof(float); break;
    case bob::io::base::Double:  base_size = sizeof(double); break;
    default:
      throw std::runtime_error("unsupported data type found while updating tensor file");
  }

  size_t tsize = 1;
  for(size_t i = 0; i < m_type.nd; ++i) tsize *= m_type.shape[i];

  m_tensor_size = tsize * base_size;
}


bob::io::base::TensorType bob::io::base::arrayTypeToTensorType(int dtype)
{
  switch(dtype)
  {
    case NPY_INT8:
      return bob::io::base::Char;
    case NPY_INT16:
      return bob::io::base::Short;
    case NPY_INT32:
      return bob::io::base::Int;
    case NPY_INT64:
      return bob::io::base::Long;
    case NPY_FLOAT32:
      return bob::io::base::Float;
    case NPY_FLOAT64:
      return bob::io::base::Double;
    default:
      throw std::runtime_error("unsupported data type found while converting array type to tensor type");
  }
}

int bob::io::base::tensorTypeToArrayType(bob::io::base::TensorType tensortype)
{
  switch(tensortype)
  {
    case bob::io::base::Char:
      return NPY_INT8;
    case bob::io::base::Short:
      return NPY_INT16;
    case bob::io::base::Int:
      return NPY_INT32;
    case bob::io::base::Long:
      return NPY_INT64;
    case bob::io::base::Float:
      return NPY_FLOAT32;
    case bob::io::base::Double:
      return NPY_FLOAT64;
    default:
      throw std::runtime_error("unsupported data type found while converting tensor type to array type");
  }
}
