#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['xbob.blitz']))
from xbob.blitz.extension import Extension

import os
package_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.join(package_dir, 'xbob', 'io', 'base', 'include')
include_dirs = [package_dir]

packages = ['bob-io >= 1.2.2']
version = '2.0.0a0'

# Check if python-imaging means pil or pillow
pil_or_pillow = []
try:
  import pkg_resources
  pkg_resources.require('PIL')
  pil_or_pillow.append('pil')
except pkg_resources.DistributionNotFound as e:
  pil_or_pillow.append('pillow')

setup(

    name='xbob.io.base',
    version=version,
    description='Base bindings for bob.io',
    url='http://github.com/bioidiap/xbob.io.base',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,

    install_requires=[
      'setuptools',
      'xbob.blitz',
    ] + pil_or_pillow,

    namespace_packages=[
      "xbob",
      "xbob.io",
      ],

    ext_modules = [
      Extension("xbob.io.base.version",
        [
          "xbob/io/base/version.cpp",
          ],
        packages = packages,
        include_dirs = include_dirs,
        version = version,
        define_macros = [('__STDC_CONSTANT_MACROS', None)],
        ),
      Extension("xbob.io.base._library",
        [
          "xbob/io/base/bobskin.cpp",
          "xbob/io/base/codec.cpp",
          "xbob/io/base/file.cpp",
          "xbob/io/base/videoreader.cpp",
          "xbob/io/base/videowriter.cpp",
          "xbob/io/base/hdf5.cpp",
          "xbob/io/base/main.cpp",
          ],
        packages = packages,
        include_dirs = include_dirs,
        version = version,
        define_macros = [('__STDC_CONSTANT_MACROS', None)],
        ),
      ],

    entry_points={
      'console_scripts': [
        'xbob_video_test.py = xbob.io.script.video_test:main',
        ],
      },

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
