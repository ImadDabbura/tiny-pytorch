"""NumPy backend implementation for NDArray operations.

This module provides a NumPy-based backend for the NDArray class, implementing
all the low-level operations needed for tensor computations. It serves as a
reference implementation and is used when the backend is set to "numpy".

The module provides functions for:
- Array creation and manipulation
- Element-wise operations
- Scalar operations
- Matrix operations
- Reduction operations

Classes
-------
Array
    Flat 1D array wrapper for NDArray storage.

Functions
---------
empty, full
    Array creation functions.
from_numpy, to_numpy
    Conversion between NumPy arrays and internal Array format.
fill, compact
    Array manipulation functions.
ewise_*, scalar_*
    Element-wise and scalar operations.
matmul
    Matrix multiplication.
reduce_*
    Reduction operations.
"""

from typing import Sequence

import numpy as np

__device_name__ = "numpy"
_datatype = np.float32
_datatype_size = np.dtype(_datatype).itemsize


class Array:
    """Flat 1D array that will be used to backup any `NDArray`.

    This class provides a simple wrapper around a numpy array that serves
    as the underlying storage for NDArray objects when using the numpy backend.

    Attributes
    ----------
    array : numpy.ndarray
        The underlying numpy array storing the data.
    """

    def __init__(self, size: int | tuple[int]):
        """Initialize a new Array with the specified size.

        Parameters
        ----------
        size : int or tuple[int]
            Size of the array.
        """
        self.array = np.empty(size, dtype=_datatype)

    @property
    def size(self) -> int:
        """Return the total number of elements in the array.

        Returns
        -------
        int
            Total number of elements in the array.
        """
        return self.array.size


def empty(shape, dtype: str | None = _datatype):
    """Create an empty array with the specified shape.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the array to create.
    dtype : str, optional
        Data type of the array. Default is float32.

    Returns
    -------
    numpy.ndarray
        Empty array with the specified shape and dtype.
    """
    return np.empty(shape, dtype)


def full(shape, fill_value, dtype: str | None = _datatype):
    """Create an array filled with a constant value.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the array to create.
    fill_value : float
        Value to fill the array with.
    dtype : str, optional
        Data type of the array. Default is float32.

    Returns
    -------
    numpy.ndarray
        Array filled with the specified value.
    """
    return np.full(shape, fill_value, dtype)


def from_numpy(array: np.ndarray, out: Array) -> None:
    """Replace the content of `out` Array with the content of `array` numpy array.

    Parameters
    ----------
    array : numpy.ndarray
        Source numpy array.
    out : Array
        Target Array to copy data into.

    Notes
    -----
    This function flattens the input array and copies its content into the
    output Array. The handler `array` will point to the newly copied array.
    """
    # Flatten returns a copy of the array
    # This means that we're copying the content of the array into out which
    # means the handler `array` will now point to the newly copied array
    out.array[:] = array.flatten()


def to_numpy(
    a: Array, shape: Sequence[int], strides: Sequence[int], offset: int
):
    """Create a view on the `a` Array.

    Parameters
    ----------
    a : Array
        Source Array.
    shape : Sequence[int]
        Shape of the view.
    strides : Sequence[int]
        Strides of the view.
    offset : int
        Offset into the array.

    Returns
    -------
    numpy.ndarray
        NumPy array view of the Array data.
    """
    return np.lib.stride_tricks.as_strided(
        a.array[offset:], shape, [s * _datatype_size for s in strides]
    )


def fill(a: Array, value):
    """Fill the array with a constant value.

    Parameters
    ----------
    a : Array
        Array to fill.
    value : float
        Value to fill the array with.
    """
    a.array.fill(value)


def compact(
    a: Array,
    out: Array,
    shape: Sequence[int],
    strides: Sequence[int],
    offset: int,
):
    """Create a compact array from a strided view.

    First creates strided view of `a` array then flatten it out to get a
    compact array that adheres to the shape/strides/offset.

    Parameters
    ----------
    a : Array
        Source Array.
    out : Array
        Output Array for the compact data.
    shape : Sequence[int]
        Shape of the view.
    strides : Sequence[int]
        Strides of the view.
    offset : int
        Offset into the array.
    """
    out.array[:] = to_numpy(a, shape, strides, offset).flatten()


def ewise_setitem(a, out, shape, strides, offset):
    """Set elements in the output array using element-wise assignment.

    Parameters
    ----------
    a : Array
        Source Array.
    out : Array
        Target Array.
    shape : tuple[int, ...]
        Shape of the view.
    strides : tuple[int, ...]
        Strides of the view.
    offset : int
        Offset into the array.
    """
    to_numpy(out, shape, strides, offset)[:] = a.array.reshape(shape)


def scalar_setitem(size, val, out, shape, strides, offset):
    """Set elements in the output array using scalar assignment.

    Parameters
    ----------
    size : int
        Size of the array.
    val : float
        Scalar value to assign.
    out : Array
        Target Array.
    shape : tuple[int, ...]
        Shape of the view.
    strides : tuple[int, ...]
        Strides of the view.
    offset : int
        Offset into the array.
    """
    to_numpy(out, shape, strides, offset)[:] = val


def ewise_add(a, b, out):
    """Element-wise addition of two arrays.

    Parameters
    ----------
    a : Array
        First input array.
    b : Array
        Second input array.
    out : Array
        Output array.
    """
    out.array[:] = a.array + b.array


def scalar_add(a, val, out):
    """Add a scalar to an array.

    Parameters
    ----------
    a : Array
        Input array.
    val : float
        Scalar value to add.
    out : Array
        Output array.
    """
    out.array[:] = a.array + val


def ewise_mul(a, b, out):
    """Element-wise multiplication of two arrays.

    Parameters
    ----------
    a : Array
        First input array.
    b : Array
        Second input array.
    out : Array
        Output array.
    """
    out.array[:] = a.array * b.array


def scalar_mul(a, val, out):
    """Multiply an array by a scalar.

    Parameters
    ----------
    a : Array
        Input array.
    val : float
        Scalar value to multiply by.
    out : Array
        Output array.
    """
    out.array[:] = a.array * val


def ewise_div(a, b, out):
    """Element-wise division of two arrays.

    Parameters
    ----------
    a : Array
        First input array (numerator).
    b : Array
        Second input array (denominator).
    out : Array
        Output array.
    """
    out.array[:] = a.array / b.array


def scalar_div(a, val, out):
    """Divide an array by a scalar.

    Parameters
    ----------
    a : Array
        Input array.
    val : float
        Scalar value to divide by.
    out : Array
        Output array.
    """
    out.array[:] = a.array / val


def scalar_power(a, val, out):
    """Raise array elements to a scalar power.

    Parameters
    ----------
    a : Array
        Input array.
    val : float
        Scalar exponent.
    out : Array
        Output array.
    """
    out.array[:] = a.array**val


def ewise_maximum(a, b, out):
    """Element-wise maximum of two arrays.

    Parameters
    ----------
    a : Array
        First input array.
    b : Array
        Second input array.
    out : Array
        Output array.
    """
    out.array[:] = np.maximum(a.array, b.array)


def scalar_maximum(a, val, out):
    """Element-wise maximum of array and scalar.

    Parameters
    ----------
    a : Array
        Input array.
    val : float
        Scalar value.
    out : Array
        Output array.
    """
    out.array[:] = np.maximum(a.array, val)


def ewise_minimum(a, b, out):
    """Element-wise minimum of two arrays.

    Parameters
    ----------
    a : Array
        First input array.
    b : Array
        Second input array.
    out : Array
        Output array.
    """
    out.array[:] = np.minimum(a.array, b.array)


def scalar_minimum(a, val, out):
    """Element-wise minimum of array and scalar.

    Parameters
    ----------
    a : Array
        Input array.
    val : float
        Scalar value.
    out : Array
        Output array.
    """
    out.array[:] = np.minimum(a.array, val)


def ewise_eq(a, b, out):
    """Element-wise equality comparison of two arrays.

    Parameters
    ----------
    a : Array
        First input array.
    b : Array
        Second input array.
    out : Array
        Output array (boolean as float32).
    """
    out.array[:] = (a.array == b.array).astype(np.float32)


def scalar_eq(a, val, out):
    """Element-wise equality comparison of array and scalar.

    Parameters
    ----------
    a : Array
        Input array.
    val : float
        Scalar value.
    out : Array
        Output array (boolean as float32).
    """
    out.array[:] = (a.array == val).astype(np.float32)


def ewise_ge(a, b, out):
    """Element-wise greater than or equal comparison of two arrays.

    Parameters
    ----------
    a : Array
        First input array.
    b : Array
        Second input array.
    out : Array
        Output array (boolean as float32).
    """
    out.array[:] = (a.array >= b.array).astype(np.float32)


def scalar_ge(a, val, out):
    """Element-wise greater than or equal comparison of array and scalar.

    Parameters
    ----------
    a : Array
        Input array.
    val : float
        Scalar value.
    out : Array
        Output array (boolean as float32).
    """
    out.array[:] = (a.array >= val).astype(np.float32)


def ewise_log(a, out):
    """Element-wise natural logarithm.

    Parameters
    ----------
    a : Array
        Input array.
    out : Array
        Output array.
    """
    out.array[:] = np.log(a.array)


def ewise_exp(a, out):
    """Element-wise exponential.

    Parameters
    ----------
    a : Array
        Input array.
    out : Array
        Output array.
    """
    out.array[:] = np.exp(a.array)


def ewise_tanh(a, out):
    """Element-wise hyperbolic tangent.

    Parameters
    ----------
    a : Array
        Input array.
    out : Array
        Output array.
    """
    out.array[:] = np.tanh(a.array)


def matmul(a, b, out, m, n, p):
    """Matrix multiplication.

    Parameters
    ----------
    a : Array
        First matrix (m x n).
    b : Array
        Second matrix (n x p).
    out : Array
        Output matrix (m x p).
    m : int
        Number of rows in first matrix.
    n : int
        Number of columns in first matrix (and rows in second).
    p : int
        Number of columns in second matrix.
    """
    out.array[:] = (a.array.reshape(m, n) @ b.array.reshape(n, p)).reshape(-1)


def reduce_sum(a, out, reduce_size):
    """Reduce sum along specified axis.

    Parameters
    ----------
    a : Array
        Input array.
    out : Array
        Output array.
    reduce_size : int
        Size of the reduction dimension.
    """
    out.array[:] = a.array[:].reshape(-1, reduce_size).sum(axis=1)


def reduce_max(a, out, reduce_size):
    """Reduce maximum along specified axis.

    Parameters
    ----------
    a : Array
        Input array.
    out : Array
        Output array.
    reduce_size : int
        Size of the reduction dimension.
    """
    out.array[:] = a.array[:].reshape(-1, reduce_size).max(axis=1)
