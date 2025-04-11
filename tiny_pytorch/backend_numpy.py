from typing import Sequence

import numpy as np


class Device:
    "Base class for devices."


class CPUDevice(Device):
    "Represents data that sits in CPU, using Numpy as backend."

    def zeros(self, shape: int | Sequence[int], dtype="float32"):
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: int | Sequence[int], dtype="float32"):
        return np.ones(shape, dtype=dtype)

    def randn(self, *shape):
        """
        Returns ndarray of shape `shape` drawn from standard normal
        distribution.
        """
        return np.random.randn(*shape)

    def rand(self, *shape):
        """
        Returns ndarray of shape `shape` drawn from uniform distribution over
        [0, 1).
        """
        return np.random.rand(*shape)

    def one_hot(self, n: int, i: int | list[int], dtype="float32"):
        """
        Returns OneHot matrix with n columns and i rows.

        Parameters
        ----------
        n: int
            Number of columns (classes).
        i: int or list of integers
            Number of onehot vectors (rows)
        """
        return np.eye(n, dtype=dtype)[i]

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __hash__(self):
        return self.__repr__().__hash__()


def cpu():
    "Returns CPU device."
    return CPUDevice()


def default_device():
    return cpu()


def all_devices():
    return [cpu()]
