from typing import Sequence

import numpy as np

__device_name__ = "numpy"
_datatype = np.float32
_datatype_size = np.dtype(_datatype).itemsize


class Array:
    """Flat 1D array that will be used to backup any `NDArray`."""

    def __init__(self, size: int | tuple[int]):
        self.array = np.empty(size, dtype=_datatype)

    @property
    def size(self) -> int:
        return self.array.size


def empty(shape, dtype: str | None = _datatype):
    return np.empty(shape, dtype)


def full(shape, fill_value, dtype: str | None = _datatype):
    return np.full(shape, fill_value, dtype)


def from_numpy(array: np.ndarray, out: Array) -> None:
    """
    Replace the content of `out` Array with the content of `array` numpy
    array.
    """
    # Flatten returns a copy of the array
    # This means that we're copying the content of the array into out which
    # means the handler `array` will now point to the newly copied array
    out.array[:] = array.flatten()


def to_numpy(
    a: Array, shape: Sequence[int], strides: Sequence[int], offset: int
):
    """Creates a view on the `a` Array."""
    return np.lib.stride_tricks.as_strided(
        a.array[offset:], shape, [s * _datatype_size for s in strides]
    )


def fill(a: Array, value):
    a.array.fill(value)


def compact(
    a: Array,
    out: Array,
    shape: Sequence[int],
    strides: Sequence[int],
    offset: int,
):
    """
    First creates strided view of `a` array then flatten it out to get a
    compact array that adheres to the shape/strides/offset.
    """
    out.array[:] = to_numpy(a, shape, strides, offset).flatten()


def ewise_setitem(a, out, shape, strides, offset):
    to_numpy(out, shape, strides, offset)[:] = a.array.reshape(shape)


def scalar_setitem(size, val, out, shape, strides, offset):
    to_numpy(out, shape, strides, offset)[:] = val


def ewise_add(a, b, out):
    out.array[:] = a.array + b.array


def scalar_add(a, val, out):
    out.array[:] = a.array + val


def ewise_mul(a, b, out):
    out.array[:] = a.array * b.array


def scalar_mul(a, val, out):
    out.array[:] = a.array * val


def ewise_div(a, b, out):
    out.array[:] = a.array / b.array


def scalar_div(a, val, out):
    out.array[:] = a.array / val


def scalar_power(a, val, out):
    out.array[:] = a.array**val


def ewise_maximum(a, b, out):
    out.array[:] = np.maximum(a.array, b.array)


def scalar_maximum(a, val, out):
    out.array[:] = np.maximum(a.array, val)


def ewise_minimum(a, b, out):
    out.array[:] = np.minimum(a.array, b.array)


def scalar_minimum(a, val, out):
    out.array[:] = np.minimum(a.array, val)


def ewise_eq(a, b, out):
    out.array[:] = (a.array == b.array).astype(np.float32)


def scalar_eq(a, val, out):
    out.array[:] = (a.array == val).astype(np.float32)


def ewise_ge(a, b, out):
    out.array[:] = (a.array >= b.array).astype(np.float32)


def scalar_ge(a, val, out):
    out.array[:] = (a.array >= val).astype(np.float32)
