#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, find_packages, dist
from distutils.extension import Extension
from distutils.version import LooseVersion

dist.Distribution(dict(setup_requires=['pypkg', 'numpy', 'blitz.array']))
import pypkg
import numpy
import blitz
import platform

# Minimum version requirements for pkg-config packages
MINIMAL_BLITZ_VERSION_REQUIRED = '0.10'
MINIMAL_BOB_VERSION_REQUIRED = '1.3'

# Pkg-config dependencies
blitz_pkg = pypkg.pkgconfig('blitz')
if blitz_pkg < MINIMAL_BLITZ_VERSION_REQUIRED:
  raise RuntimeError("This package requires Blitz++ %s or superior, but you have %s" % (MINIMAL_BLITZ_VERSION_REQUIRED, blitz_pkg.version))

bob_pkg = pypkg.pkgconfig('bob-io')
if bob_pkg < MINIMAL_BOB_VERSION_REQUIRED:
  raise RuntimeError("This package requires Bob %s or superior, but you have %s" % (MINIMAL_BOB_VERSION_REQUIRED, bob_pkg.version))

# Make-up the names of versioned Bob libraries we must link against
if platform.system() == 'Darwin':
  bob_libraries=['%s.%s' % (k, bob_pkg.version) for k in bob_pkg.libraries()]
elif platform.system() == 'Linux':
  bob_libraries=[':lib%s.so.%s' % (k, bob_pkg.version) for k in bob_pkg.libraries()]
else:
  raise RuntimeError("This package currently only supports MacOSX and Linux builds")

# Add system include directories
extra_compile_args = []
system_includes = \
    [blitz.get_include()] + \
    blitz_pkg.include_directories() + \
    [numpy.get_include()]
for k in system_includes: extra_compile_args += ['-isystem', k]

# NumPy API macros necessary?
define_macros=[
    ("PY_ARRAY_UNIQUE_SYMBOL", "XBOB_IO_PY_ARRAY_API"),
    ("NO_IMPORT_ARRAY", "1"),
    ]

import numpy
from distutils.version import StrictVersion
if StrictVersion(numpy.__version__) >= StrictVersion('1.7'):
  define_macros.append(("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"))

# Compilation options
if platform.system() == 'Darwin':
  extra_compile_args += ['-std=c++11', '-Wno-#warnings']
else:
  extra_compile_args += ['-std=c++11']

# Local include directory
import os
package_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.join(package_dir, 'xbob', 'io', 'include')
include_dirs = [package_dir]

# Define package version
version = '2.0.0a0'
define_macros += [
    ("XBOB_IO_VERSION", version),
    ]

setup(

    name='xbob.io',
    version=version,
    description='Bindings for bob.io',
    url='http://github.com/anjos/xbob.io',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,

    install_requires=[
      'setuptools',
      'blitz.array',
    ],

    namespace_packages=[
      "xbob",
      ],

    ext_modules = [
      Extension("xbob.io._library",
        [
          "xbob/io/main.cpp",
          ],
        define_macros=define_macros,
        include_dirs=include_dirs + bob_pkg.include_directories(),
        extra_compile_args=extra_compile_args,
        library_dirs=bob_pkg.library_directories(),
        runtime_library_dirs=bob_pkg.library_directories(),
        libraries=bob_libraries,
        language="c++",
        ),
      ],

    classifiers = [
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Software Development :: Libraries :: Python Modules',
      ],

    )
