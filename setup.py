#!/usr/bin/env python3
"""
Setup script for tiny-pytorch with C++ extensions.
"""

import os
import sys

import pybind11
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


def get_compiler_flags():
    """Get compiler flags for C++ extensions."""
    flags = []

    # Add C++11 standard and optimization
    if sys.platform.startswith("linux"):
        flags.extend(["-std=c++11", "-O3", "-fPIC", "-march=native"])
    elif sys.platform == "darwin":  # macOS
        flags.extend(["-std=c++11", "-O3", "-march=native"])
    elif sys.platform == "win32":  # Windows
        flags.extend(["/std:c++11", "/O2"])

    return flags


def get_linker_flags():
    """Get linker flags for C++ extensions."""
    flags = []

    if sys.platform.startswith("linux"):
        flags.extend(["-shared"])
    elif sys.platform == "darwin":  # macOS
        flags.extend(["-bundle", "-undefined", "dynamic_lookup"])

    return flags


class BuildExt(build_ext):
    """Custom build command for C++ extensions."""

    def build_extension(self, ext):
        # Add compiler flags
        if hasattr(self.compiler, "compiler_so"):
            self.compiler.compiler_so.extend(get_compiler_flags())

        # Add linker flags
        if hasattr(self.compiler, "linker_so"):
            self.compiler.linker_so.extend(get_linker_flags())

        # Call the original build_extension method
        super().build_extension(ext)


# Read the README file
def read_readme():
    with open("README.md", encoding="utf-8") as fh:
        return fh.read()


# Define C++ extensions
extensions = [
    Extension(
        "tiny_pytorch.ndarray_backend_cpu",
        sources=["tiny_pytorch/ndarray_backend_cpu.cpp"],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
        ],
        language="c++",
        extra_compile_args=get_compiler_flags(),
        extra_link_args=get_linker_flags(),
    ),
]


if __name__ == "__main__":
    setup(
        name="tiny-pytorch",
        version="0.0.4",
        author="Imad Dabbura",
        author_email="imad.dabbura@hotmail.com",
        description="Mini Deep Learning framework similar to PyTorch",
        long_description=read_readme(),
        long_description_content_type="text/markdown",
        packages=find_packages(),
        ext_modules=extensions,
        cmdclass={"build_ext": BuildExt},
        install_requires=["numpy>=2.2.3,<3.0.0"],
        python_requires=">=3.13",
        zip_safe=False,  # Required for C++ extensions
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: Apache License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.13",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )
