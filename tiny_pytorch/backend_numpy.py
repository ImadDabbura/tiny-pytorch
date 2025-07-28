"""NumPy backend implementation for tiny-pytorch.

This module provides a NumPy-based backend for tiny-pytorch, implementing
device abstractions and tensor operations using NumPy arrays. It serves as
a simple, pure Python backend that's useful for development, testing, and
educational purposes.

The module provides a complete device abstraction layer that wraps NumPy
functionality, making it easy to create and manipulate tensors using
standard NumPy operations. This backend is particularly useful when you
want to avoid external dependencies or need a simple, portable implementation.

Key Features
-----------
- Pure Python implementation using NumPy
- Device abstraction for tensor operations
- Standard tensor creation methods (zeros, ones, randn, rand)
- One-hot encoding support
- Uniform interface with other backends
- No external dependencies beyond NumPy

Classes
-------
Device
    Base class for device abstractions.
    Defines the interface that all device implementations must follow.
    Devices represent where tensor data is stored and provide methods
    for creating tensors on that device.
CPUDevice
    CPU device implementation using NumPy arrays.
    Represents data that sits in CPU memory, using NumPy as the underlying
    computational engine. Provides methods for creating tensors with
    different initializations and distributions.

Functions
---------
cpu() -> CPUDevice
    Returns a CPU device instance using NumPy backend.
default_device() -> CPUDevice
    Returns the default device (CPU with NumPy backend).
all_devices() -> list[CPUDevice]
    Returns a list of all available devices (only CPU for this backend).

Notes
-----
This backend implementation is designed to be simple and portable, making
it ideal for development, testing, and educational purposes. It provides
the same interface as other backends (CPU, CUDA) but uses NumPy arrays
as the underlying storage and computation engine.

The backend automatically handles data type conversions and provides
consistent behavior across different platforms. All tensor operations
are performed using NumPy's optimized C implementations, ensuring
good performance for most use cases.

This backend is particularly useful when:
- You need a simple, portable implementation
- You want to avoid external dependencies
- You're developing or testing code
- You're in an educational environment

Examples
--------
>>> import tiny_pytorch as tp
>>>
>>> # Create a CPU device using NumPy backend
>>> device = tp.backend_numpy.cpu()
>>>
>>> # Create tensors on the device
>>> x = device.zeros((3, 4), dtype="float32")
>>> y = device.ones((3, 4), dtype="float32")
>>> z = device.randn(3, 4)  # Random normal distribution
>>> w = device.rand(3, 4)   # Random uniform distribution
>>>
>>> # Create one-hot encoded tensor
>>> one_hot = device.one_hot(10, 5, dtype="float32")
>>>
>>> # Use the default device
>>> default_dev = tp.backend_numpy.default_device()
>>> tensor = default_dev.full((2, 2), 3.14, dtype="float32")
>>>
>>> # Check available devices
>>> devices = tp.backend_numpy.all_devices()
>>> print(f"Available devices: {devices}")
"""

from typing import Sequence

import numpy as np


class Device:
    """Base class for device abstractions.

    This class defines the interface that all device implementations
    must follow. Devices represent where tensor data is stored and
    provide methods for creating tensors on that device.
    """


class CPUDevice(Device):
    """Represents data that sits in CPU, using NumPy as backend.

    This device implementation uses NumPy arrays for all tensor operations.
    It provides methods for creating tensors with different initializations
    and distributions.

    Methods
    -------
    zeros(shape, dtype)
        Create a tensor filled with zeros.
    ones(shape, dtype)
        Create a tensor filled with ones.
    randn(*shape)
        Create a tensor with random values from standard normal distribution.
    rand(*shape)
        Create a tensor with random values from uniform distribution.
    one_hot(n, i, dtype)
        Create a one-hot encoded tensor.
    """

    def zeros(self, shape: int | Sequence[int], dtype="float32"):
        """Create a tensor filled with zeros.

        Parameters
        ----------
        shape : int or Sequence[int]
            The shape of the tensor to create.
        dtype : str, optional
            The data type of the tensor. Default is "float32".

        Returns
        -------
        numpy.ndarray
            A tensor filled with zeros.
        """
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: int | Sequence[int], dtype="float32"):
        """Create a tensor filled with ones.

        Parameters
        ----------
        shape : int or Sequence[int]
            The shape of the tensor to create.
        dtype : str, optional
            The data type of the tensor. Default is "float32".

        Returns
        -------
        numpy.ndarray
            A tensor filled with ones.
        """
        return np.ones(shape, dtype=dtype)

    def randn(self, *shape):
        """Create a tensor with random values from standard normal distribution.

        Parameters
        ----------
        *shape : int
            The shape of the tensor to create.

        Returns
        -------
        numpy.ndarray
            A tensor with random values from N(0, 1).
        """
        return np.random.randn(*shape)

    def rand(self, *shape):
        """Create a tensor with random values from uniform distribution.

        Parameters
        ----------
        *shape : int
            The shape of the tensor to create.

        Returns
        -------
        numpy.ndarray
            A tensor with random values from U[0, 1).
        """
        return np.random.rand(*shape)

    def one_hot(self, n: int, i: int | list[int], dtype="float32"):
        """Create a one-hot encoded tensor.

        Parameters
        ----------
        n : int
            Number of columns (classes).
        i : int or list[int]
            Number of one-hot vectors (rows) or indices to encode.
        dtype : str, optional
            The data type of the tensor. Default is "float32".

        Returns
        -------
        numpy.ndarray
            A one-hot encoded tensor.
        """
        return np.eye(n, dtype=dtype)[i]

    def full(
        self,
        shape: int | Sequence[int],
        fill_value: float,
        dtype: str = "float32",
    ) -> np.ndarray:
        """Create a tensor filled with a constant value.

        Parameters
        ----------
        shape : int or Sequence[int]
            The shape of the tensor to create.
        fill_value : float
            The value to fill the tensor with.
        dtype : str, optional
            The data type of the tensor. Default is "float32".

        Returns
        -------
        numpy.ndarray
            A tensor filled with the specified value.
        """
        return np.full(shape, fill_value, dtype=dtype)

    def __eq__(self, other):
        """Check if this device equals another device."""
        return isinstance(other, self.__class__)

    def __repr__(self):
        """String representation of the device."""
        return f"{self.__class__.__name__}()"

    def __hash__(self):
        """Hash of the device."""
        return self.__repr__().__hash__()


def cpu():
    """Returns a CPU device instance.

    Returns
    -------
    CPUDevice
        A CPU device instance.
    """
    return CPUDevice()


def default_device():
    """Returns the default device (CPU).

    Returns
    -------
    CPUDevice
        The default CPU device.
    """
    return cpu()


def all_devices():
    """Returns a list of all available devices.

    Returns
    -------
    list[CPUDevice]
        A list containing the CPU device.
    """
    return [cpu()]
