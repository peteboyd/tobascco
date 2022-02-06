# -*- coding: utf-8 -*-
import os

from setuptools import find_packages, setup
from numpy.distutils.core import Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

import versioneer

include_dirs = [os.path.join(os.getcwd(), "tobascco", "src")]


with open("requirements.txt", "r") as fh:
    REQUIREMENTS = fh.readlines()


with open("README.md", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

setup(
    name="tobascco",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Assembles MOFs",
    setup_requires=["numpy"],
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["tobascco", "tobascco.*"]),
    url="https://github.com/peteboyd/tobascco",
    license="Apache 2.0",
    install_requires=REQUIREMENTS,
    extras_require={
        "testing": ["pytest>=6,<8", "pytest-cov>=2,<4"],
        "docs": [
            "sphinx>=3,<5",
            "sphinx-book-theme==0.*",
            "sphinx-autodoc-typehints==1.*",
            "sphinx-copybutton==0.*",
        ],
        "pre-commit": [
            "pre-commit==2.*",
            "pylint==2.*",
            "isort==5.*",
        ],
        "dev": [
            "versioneer==0.*",
            "black>=20,<23",
        ],
    },
    author="Peter Boyd",
    author_email="peter.g.boyd@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=[
        Extension(
            "_nloptimize",
            include_dirs=include_dirs + get_numpy_include_dirs(),
            sources=[os.path.join(os.getcwd(), "tobascco", "src", "pyoptim.cpp")],
            language="c++",
            libraries=["nlopt"],
            extra_link_args=["-O"],
        )
     ],
)
