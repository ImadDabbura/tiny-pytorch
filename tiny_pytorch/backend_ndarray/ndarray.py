from math import prod

import numpy as np

from . import ndarray_backend_numpy


class BackendDevice:
    """Backend devive that wraps the implementation module for each device."""

    def __init__(self, name: str, module=None):
        self.name = name
        # Key attribute that will handle all ops on its device
        self.module = module

    def __eq__(self, other):
        # Two devives are equal if they have the same name
        return self.name == other.name

    def __repr__(self):
        return f"{self.name}()"

    def enabled(self):
        return self.module is not None

    def __getattr__(self, name):
        # All attempts to get attribute from device will be forwarded to the
        # module that implements the device's operations
        # i.e. device.op will become self.module.op
        return getattr(self.module, name)


def cpu_numpy():
    return BackendDevice("cpu_numpy", ndarray_backend_numpy)


def default_device():
    """Return cpu numpy backend."""
    return cpu_numpy()


class NDArray:
    """
    A generic ND array class that may contain multipe different backends
    i.e., a Numpy backend, a native CPU backend, or a GPU backend.

    For now, for simplicity the class only supports float32 types,
    though this can be extended if desired.
    """

    def __init__(self, other, device=None):
        """
        Create NDArray by copying another NDArray/Numpy array or create
        Or use Numpy as a bridge for all other types of iterables.
        """
        if isinstance(other, NDArray):
            if device is None:
                device = other.device
            # Creates a copy because any ops will create new array
            self._init(other.to(device) + 0.0)
        elif isinstance(other, np.ndarray):
            device = device if device is not None else default_device()
            array = NDArray.make(other.shape, device=device)
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
        else:
            # Check if we can create a numpy array from input
            array = NDArray(np.array(other), device=device)
            self._init(array)

    def _init(self, other: "NDArray"):
        # shape, stride, and offset allows us to represent 1D array as NDarray
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        # BackendDevice: helps us dispatch ops to the corresponding device
        self._device = other._device
        self._handle = other._handle  # pointer to 1D array

    def to(self, device):
        if device == self.device:
            return self
        # Use Numpy as a bridge by first converting NDArray to numpy array first
        return NDArray(self.numpy(), device)

    def numpy(self):
        return self.device.to_numpy(
            self._handle, self.shape, self.strides, self._offset
        )

    @staticmethod
    def compact_strides(shape):
        """Utility function to compute compact strides."""
        # ND-dimensional arrays are represented (with row-major order)
        # contiguously from the inner most dimension to the outer most
        # dimension.
        # Examples:
        #   1. 4 x 3 array will be represented physically in memory with first
        #   row (3 elements) then the second and so on
        #   2. 4 x 3 x 2 array will be represented with inner most dimension
        #   first first until its done (2 in this case), then next outer dimension
        #   (3 rows of 2), finally outer most dimension which has 4 (3 x 2)
        #   arrays
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])

    @staticmethod
    def make(shape, strides=None, device=None, handle=None, offset=0):
        """
        Create a new NDArray with the given properties.  This will allocate the
        memory if handle is None, otherwise it will use the handle of an existing array.
        """
        array = NDArray.__new__(NDArray)
        array._shape = tuple(shape)
        array._strides = (
            NDArray.compact_strides(shape) if strides is None else strides
        )
        array._offset = offset
        array._device = device if device is not None else default_device()
        if handle is None:
            # Number of elements = product of shape all devices allocate memory
            # of size prod(shape) and return the array created as the handler
            # because All NDArray objects underlying memory is just flat 1D array
            array._handle = array.device.Array(prod(shape))
        else:
            array._handle = handle
        return array

    @property
    def device(self):
        return self._device

    @property
    def shape(self):
        return self._shape

    @property
    def strides(self):
        return self._strides

    @property
    def size(self):
        return prod(self._shape)

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def dtype(self):
        # Only supporting float32 for now
        return "float32"

    def __str__(self):
        return str(self.numpy())

    def __repr__(self):
        return f"NDArray({str(self)}, device={self.device})"

    def fill(self, value):
        """Fill in-place with a constant value."""
        self._device.fill(self._handle, value)
