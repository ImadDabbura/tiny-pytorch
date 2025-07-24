"""NumPy backend implementation for tiny-pytorch.

This module provides a NumPy-based backend for tiny-pytorch, implementing
device abstractions and tensor operations using NumPy arrays. It serves as
a simple, pure Python backend that's useful for development and testing.

Classes
-------
Device
    Base class for device abstractions.
CPUDevice
    CPU device implementation using NumPy arrays.

Functions
---------
cpu
    Returns a CPU device instance.
default_device
    Returns the default device (CPU).
all_devices
    Returns a list of all available devices.
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
