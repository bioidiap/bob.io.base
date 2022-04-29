#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST
#
from setuptools import dist, setup

dist.Distribution(dict(setup_requires=["bob.extension"]))

from bob.extension.utils import find_packages, load_requirements

install_requires = load_requirements()

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(
    # This is the basic information about your project. Modify all this
    # information before releasing code publicly.
    name="bob.io.base",
    version=open("version.txt").read().rstrip(),
    description="Basic IO for Bob",
    url="http://gitlab.idiap.ch/bob/bob.io.base",
    license="BSD",
    author="Andre Anjos",
    author_email="andre.anjos@idiap.ch",
    long_description=open("README.rst").read(),
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    setup_requires=install_requires,
    install_requires=install_requires,
    # Classifiers are important if you plan to distribute this package through
    # PyPI. You can find the complete list of classifiers that are valid and
    # useful here (http://pypi.python.org/pypi?%3Aaction=list_classifiers).
    classifiers=[
        "Framework :: Bob",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
