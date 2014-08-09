/**
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This class can be used to store and load multiarrays into/from files.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#define BOB_IO_BASE_MODULE
#include <bob.io.base/api.h>
#include "TensorFile.h"

bob::io::base::TensorFile::TensorFile(const std::string& filename,
    bob::io::base::TensorFile::openmode flag):
  m_header_init(false),
  m_current_array(0),
  m_n_arrays_written(0),
  m_openmode(flag)
{
  if((flag & bob::io::base::TensorFile::out) && (flag & bob::io::base::TensorFile::in)) {
    m_stream.open(filename.c_str(), std::ios::in | std::ios::out |
        std::ios::binary);
    if(m_stream)
    {
      m_header.read(m_stream);
      m_buffer.reset(new char[BobIoTypeinfo_BufferSize(&m_header.m_type)]);
      m_header_init = true;
      m_n_arrays_written = m_header.m_n_samples;

      if (flag & bob::io::base::TensorFile::append) {
        m_stream.seekp(0, std::ios::end);
        m_current_array = m_header.m_n_samples;
      }
    }
  }
  else if(flag & bob::io::base::TensorFile::out) {
    if(m_stream && (flag & bob::io::base::TensorFile::append)) {
      m_stream.open(filename.c_str(), std::ios::out | std::ios::in |
          std::ios::binary);
      m_header.read(m_stream);
      m_buffer.reset(new char[BobIoTypeinfo_BufferSize(&m_header.m_type)]);
      m_header_init = true;
      m_n_arrays_written = m_header.m_n_samples;
      m_stream.seekp(0, std::ios::end);
      m_current_array = m_header.m_n_samples;
    }
    else
      m_stream.open(filename.c_str(), std::ios::out | std::ios::binary);
  }
  else if(flag & bob::io::base::TensorFile::in) {
    m_stream.open(filename.c_str(), std::ios::in | std::ios::binary);
    if(m_stream) {
      m_header.read(m_stream);
      m_buffer.reset(new char[BobIoTypeinfo_BufferSize(&m_header.m_type)]);
      m_header_init = true;
      m_n_arrays_written = m_header.m_n_samples;

      if (flag & bob::io::base::TensorFile::append) {
        throw std::runtime_error("cannot append data in read only mode");
      }
    }
  }
  else {
    throw std::runtime_error("invalid combination of flags");
  }
}

bob::io::base::TensorFile::~TensorFile() {
  close();
}

void bob::io::base::TensorFile::peek(BobIoTypeinfo& info) const {
  info = m_header.m_type;
}

void bob::io::base::TensorFile::close() {
  // Rewrite the header and update the number of samples
  m_header.m_n_samples = m_n_arrays_written;
  if(m_openmode & bob::io::base::TensorFile::out) m_header.write(m_stream);

  m_stream.close();
}

void bob::io::base::TensorFile::initHeader(const BobIoTypeinfo& info) {
  // Check that data have not already been written
  if (m_n_arrays_written > 0 ) {
    throw std::runtime_error("cannot init the header of an output stream in which data have already been written");
  }

  // Initialize header
  m_header.m_type = info;
  m_header.m_tensor_type = bob::io::base::arrayTypeToTensorType(info.dtype);
  m_header.write(m_stream);

  // Temporary buffer to help with data transposition...
  m_buffer.reset(new char[BobIoTypeinfo_BufferSize(&m_header.m_type)]);

  m_header_init = true;
}

void bob::io::base::TensorFile::write(const bob::io::base::array::interface& data) {

  const BobIoTypeinfo& info = data.type();

  if (!m_header_init) initHeader(info);
  else {
    //checks compatibility with previously written stuff
    if (!BobIoTypeinfo_IsCompatible(&m_header.m_type, &info))
      throw std::runtime_error("buffer does not conform to expected type");
  }

  BobIoReorder_RowToCol(data.ptr(), m_buffer.get(), &info);

  m_stream.write(static_cast<const char*>(m_buffer.get()), BobIoTypeinfo_BufferSize(&info));

  // increment m_n_arrays_written and m_current_array
  ++m_current_array;
  if (m_current_array>m_n_arrays_written) ++m_n_arrays_written;
}

void bob::io::base::TensorFile::read (bob::io::base::array::interface& buf) {

  if(!m_header_init) {
    throw std::runtime_error("TensorFile: header is not initialized");
  }
  if(!BobIoTypeinfo_IsCompatible(&buf.type(), &m_header.m_type)) buf.set(m_header.m_type);

  m_stream.read(reinterpret_cast<char*>(m_buffer.get()),
      BobIoTypeinfo_BufferSize(&m_header.m_type));

  BobIoReorder_ColToRow(m_buffer.get(), buf.ptr(), &m_header.m_type);

  ++m_current_array;
}

void bob::io::base::TensorFile::read (size_t index, bob::io::base::array::interface& buf) {

  // Check that we are reaching an existing array
  if( index > m_header.m_n_samples ) {
    boost::format m("request to read list item at position %d which is outside the bounds of declared object with size %d");
    m % index % m_header.m_n_samples;
    throw std::runtime_error(m.str());
  }

  // Set the stream pointer at the correct position
  m_stream.seekg( m_header.getArrayIndex(index) );
  m_current_array = index;

  // Put the content of the stream in the blitz array.
  read(buf);
}
