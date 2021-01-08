#!/usr/bin/env python

from numpy.distutils.core import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs
import os

# NB for shared libraries ensure that the lib dir is in LD_LIBRARY_PATH!
# eg export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/pboyd/lib/nlopt-2.4.1/lib
# os.environ["CC"] = "g++"
# os.environ["CXX"] = "g++"
include_dirs = [os.getcwd(), os.environ["NL_INCDIR"]]

module = Extension(
    "_nloptimize",
    include_dirs=include_dirs + get_numpy_include_dirs(),
    sources=["pyoptim.cpp"],
    language="c++",
    libraries=["nlopt"],
    library_dirs=[os.environ["NL_LIBDIR"]],
    extra_link_args=["-O"],
)
# NB: add "-g" to the extra_link_args list if debugging is required

setup(name="nloptimize", description="Package for ...", ext_modules=[module])
