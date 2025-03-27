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


def from_numpy(array: np.ndarray, out: Array) -> None:
    """
    Replace the content of `out` Array with the content of `array` numpy
    array.
    """
    # Flatten returns a copy of the array
    # This means that we're copying the content of the array into out which
    # means the handler `array` will now point to the newly copied array
    out.array[:] = array.flatten()
