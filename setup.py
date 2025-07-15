#!/usr/bin/env python3
"""
Setup script for tiny-pytorch with C++ and CUDA extensions.
"""

import os
import subprocess
import sys
from pathlib import Path

import pybind11
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


def get_compiler_flags():
    """Get compiler flags for C++ extensions."""
    flags = []

    # Add C++11 standard and optimization
    if sys.platform.startswith("linux"):
        # For Linux, use conservative flags that work across different architectures
        flags.extend(["-std=c++11", "-O3", "-fPIC"])
        # Don't use architecture-specific flags to avoid compatibility issues
    elif sys.platform == "darwin":  # macOS
        # For macOS, use conservative flags that work on both Intel and Apple Silicon
        flags.extend(["-std=c++11", "-O3"])
        # Don't use architecture-specific flags to avoid compatibility issues
    elif sys.platform == "win32":  # Windows
        # For Windows, use conservative flags that work across different architectures
        flags.extend(["/std:c++11", "/O2"])
        # Don't use architecture-specific flags to avoid compatibility issues

    return flags


def get_linker_flags():
    """Get linker flags for C++ extensions."""
    flags = []

    if sys.platform.startswith("linux"):
        flags.extend(["-shared"])
    elif sys.platform == "darwin":  # macOS
        flags.extend(["-bundle", "-undefined", "dynamic_lookup"])

    return flags


def get_cuda_compiler_flags():
    """Get CUDA compiler flags."""
    flags = []

    # Basic CUDA flags
    flags.extend(["-std=c++11", "-O3"])

    # Architecture flags - use compute capability 6.0 as minimum
    # This covers most modern GPUs from 2016 onwards
    flags.extend(["-arch=sm_60"])

    # Enable fast math
    flags.extend(["--use_fast_math"])

    # Generate position independent code
    if sys.platform.startswith("linux"):
        flags.extend(["-Xcompiler", "-fPIC"])

    return flags


def find_cuda():
    """Find CUDA installation and return paths."""
    # Try to find CUDA installation
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")

    if not cuda_home:
        # Try common installation paths
        common_paths = [
            "/usr/local/cuda",
            "/opt/cuda",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA",
        ]

        for path in common_paths:
            if os.path.exists(path):
                # Find the latest version
                if os.path.isdir(path):
                    versions = [
                        d for d in os.listdir(path) if d.startswith("v")
                    ]
                    if versions:
                        latest_version = sorted(versions)[-1]
                        cuda_home = os.path.join(path, latest_version)
                        break
                else:
                    cuda_home = path
                    break

    if not cuda_home or not os.path.exists(cuda_home):
        return None, None, None

    cuda_include = os.path.join(cuda_home, "include")
    cuda_lib = (
        os.path.join(cuda_home, "lib64")
        if os.path.exists(os.path.join(cuda_home, "lib64"))
        else os.path.join(cuda_home, "lib")
    )

    return cuda_home, cuda_include, cuda_lib


def get_cuda_libraries():
    """Get CUDA libraries to link against."""
    return ["cudart"]


class CUDABuildExt(build_ext):
    """Custom build command for C++ and CUDA extensions."""

    def build_extension(self, ext):
        # Check if this is a CUDA extension
        if ext.language == "cuda":
            self._build_cuda_extension(ext)
        else:
            # Handle regular C++ extensions
            if hasattr(self.compiler, "compiler_so"):
                self.compiler.compiler_so.extend(get_compiler_flags())
            if hasattr(self.compiler, "linker_so"):
                self.compiler.linker_so.extend(get_linker_flags())
            super().build_extension(ext)

    def _build_cuda_extension(self, ext):
        """Build CUDA extension using nvcc."""
        cuda_home, cuda_include, cuda_lib = find_cuda()
        if not cuda_home:
            raise RuntimeError("CUDA not found. Cannot build CUDA extension.")

        # Find nvcc compiler
        nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
        if not os.path.exists(nvcc_path):
            raise RuntimeError(f"nvcc not found at {nvcc_path}")

        # Create build directory
        build_dir = os.path.join(self.build_lib, "tiny_pytorch")
        os.makedirs(build_dir, exist_ok=True)

        # Determine output file
        if sys.platform.startswith("linux"):
            ext_name = "ndarray_backend_cuda.so"
        elif sys.platform == "darwin":
            ext_name = "ndarray_backend_cuda.so"
        elif sys.platform == "win32":
            ext_name = "ndarray_backend_cuda.pyd"
        else:
            raise RuntimeError(f"Unsupported platform: {sys.platform}")

        output_file = os.path.join(build_dir, ext_name)

        # Build command
        cmd = [
            nvcc_path,
            *get_cuda_compiler_flags(),
            "-I",
            pybind11.get_include(),
            "-I",
            pybind11.get_include(user=True),
            "-I",
            cuda_include,
            "--shared",
            "-o",
            output_file,
            ext.sources[0],  # The .cu file
        ]

        # Add library paths and libraries
        cmd.extend(["-L", cuda_lib])
        for lib in ext.libraries:
            cmd.extend(["-l", lib])

        # Add linker flags
        if sys.platform.startswith("linux"):
            cmd.extend(["-Xlinker", "-fPIC"])
        elif sys.platform == "darwin":
            cmd.extend(
                [
                    "-Xlinker",
                    "-bundle",
                    "-Xlinker",
                    "-undefined",
                    "-Xlinker",
                    "dynamic_lookup",
                ]
            )

        print("Building CUDA extension: " + " ".join(cmd))

        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("CUDA compilation failed:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            raise RuntimeError("CUDA compilation failed")

        print(f"CUDA extension built successfully: {output_file}")


# Read the README file
def read_readme():
    with open("README.md", encoding="utf-8") as fh:
        return fh.read()


# Find CUDA installation
cuda_home, cuda_include, cuda_lib = find_cuda()

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

# Add CUDA extension if CUDA is available
if cuda_home and cuda_include and cuda_lib:
    print(f"Found CUDA installation at: {cuda_home}")
    extensions.append(
        Extension(
            "tiny_pytorch.ndarray_backend_cuda",
            sources=["tiny_pytorch/ndarray_backend_cuda.cu"],
            include_dirs=[
                pybind11.get_include(),
                pybind11.get_include(user=True),
                cuda_include,
            ],
            library_dirs=[cuda_lib],
            libraries=get_cuda_libraries(),
            language="cuda",
            extra_compile_args=get_cuda_compiler_flags(),
            extra_link_args=get_linker_flags(),
        )
    )
else:
    print("CUDA not found. CUDA backend will not be available.")
    print(
        "To enable CUDA support, install CUDA and set CUDA_HOME environment variable."
    )


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
        cmdclass={"build_ext": CUDABuildExt},
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
