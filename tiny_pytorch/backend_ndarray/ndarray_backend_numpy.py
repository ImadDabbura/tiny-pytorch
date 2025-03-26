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
