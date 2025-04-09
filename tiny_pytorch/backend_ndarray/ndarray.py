from math import prod
from numbers import Number

import numpy as np

from ..utils import tuplify
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

    NDArray is basically a Python wrapper for handling operations on
    n-dimensional arrays. The underlying array is just a flat 1D array and the
    backend device will handle all the ops on the 1D array. Strides, shape, and
    offset allows us to map n-dimensional (logical) array to the 1D flat array
    that are physically allocated in memory.
    The high level ops such as broadcasting and transposing are all done in
    Python without touching the underlying array.
    The other `raw` ops such as addition and matrix-multiplication will be
    implemented is C/C++ that would call highly optimized Kernels for such ops
    such as CUDA Kernels.
    To make the backend code simpler, we will only operate on compact arrays so
    we need to call `compact()` before any ops AND we support only float32 data
    type.
    """

    def __init__(self, other, device=None):
        """
        Create NDArray by copying another NDArray/Numpy array OR use Numpy as
        a bridge for all other types of iterables.
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
        self._handle = other._handle  # pointer to 1D array of type `Array`

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
        """
        Utility function to compute compact strides.

        N-dimensional arrays are represented (with row-major order)
        contiguously from the inner most dimension to the outer most
        dimension.
        Examples:
          1. 4 x 3 array will be represented physically in memory with first
          row (3 elements) then the second and so on -> strides = (3, 1)
          2. 4 x 3 x 2 array will be represented with inner most dimension
          first until its done (2 in this case), then next outer dimension
          (3 rows of 2), finally outer most dimension which has 4 (3 x 2)
          arrays -> strides = (6, 2, 1)
        """
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])

    @staticmethod
    def make(shape, strides=None, device=None, handle=None, offset=0):
        """
        Create a new NDArray with the given properties. Memory will only be
        allocated if `handle` is None, otherwise it will use the same
        underlying memory.
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

    def is_compact(self):
        """
        Return true if array is compact in memory and internal size equals
        product of the shape dimensions.
        """
        # An array is contiguous is it has the same strides as row-major order
        # and offset = 0 (i.e. same size as the original array)
        return self._strides == NDArray.compact_strides(
            self._shape
        ) and self._handle.size == prod(self._shape)

    def compact(self):
        """Convert NDArray to be compact if it is not already compact."""
        if self.is_compact():
            return self
        out = NDArray.make(self._shape, device=self._device)
        self.device.compact(
            self._handle, out._handle, self._shape, self._strides, self._offset
        )
        return out

    def as_strided(self, shape, strides):
        """
        Create a strided view of the underlying memory without copying anything.
        """
        assert len(shape) == len(strides)
        return NDArray.make(shape, strides, self._device, self._handle)

    def reshape(self, new_shape):
        """
        Reshape the matrix without copying memory.  This will return a matrix
        that corresponds to a reshaped array but points to the same memory as
        the original array. Therefore, we only change the shape and the strides
        to get the new n-dimensional logical view of the array.

        Parameters
        ----------
        new_shape: tuple
            New shape of the array.

        Returns
        -------
        NDArray
            Reshaped array; this will point to the same memory as the original
            NDArray

        Raises
        ------
        ValueError
            If product of current shape is not equal to the product of the new
            shape, or if the matrix is not compact.
        """
        if prod(self._shape) != prod(new_shape) or not self.is_compact():
            raise ValueError()
        new_strides = self.compact_strides(new_shape)
        return self.make(
            new_shape, new_strides, self._device, self._handle, self._offset
        )

    def permute(self, new_axes):
        """
        Permute order of the dimensions. `new_axes` describes a permutation of
        the existing axes, Example:

          - If we have an array with dimension "BHWC" then
            `.permute((0,3,1,2))`
            would convert this to "BCHW" order.
          - For a 2D array, `.permute((1,0))` would transpose the array.

        Like `reshape`, this operation should not copy memory, but achieves the
        permuting by just adjusting the shape/strides of the array.  That is,
        it returns a new array that has the dimensions permuted as desired, but
        which points to the same memory as the original array.

        Parameters
        ----------
        new_axes: tuple
            Permutation order of the dimensions

        Returns
        -------
        NDarray
            New NDArray object with permuted dimensions, pointing to the same
            memory as the original NDArray (i.e., just shape and strides changed).
        """
        assert len(self._shape) == len(new_axes)
        shape = tuple(self._shape[i] for i in new_axes)
        strides = tuple(self._strides[i] for i in new_axes)
        return self.make(
            shape, strides, self._device, self._handle, self._offset
        )

    def broadcast_to(self, new_shape):
        """
        Broadcast an array to a new shape.  `new_shape`'s elements must be the
        same as the original shape, except for dimensions in the self where
        the size = 1 (which can then be broadcast to any size). This will not
        copy memory, and just achieves broadcasting by manipulating the strides.

        Parameters
        ----------
        new_shape: tuple
            Shape to broadcast to

        Returns
        -------
        NDArray:
            New NDArray object with the new broadcast shape; should
            point to the same memory as the original array.

        Raises
        ------
        AssertionError
            If new_shape[i] != shape[i] for all i where shape[i] != 1
        """
        assert len(self._shape) == len(new_shape)
        assert all(
            e == new_shape[i] for i, e in enumerate(self._shape) if e != 1
        )
        new_strides = tuple(
            0 if e == 1 else self._strides[i]
            for i, e in enumerate(self._shape)
        )
        return self.make(
            new_shape, new_strides, self._device, self._handle, self._offset
        )

    def _process_slice(self, sl, dim):
        """Convert a slice to an explicit start/stop/step"""
        start, stop, step = sl.start, sl.stop, sl.step
        if start is None:
            start = 0
        elif start < 0:
            start += self.shape[dim]
        if stop is None:
            stop = self.shape[dim]
        elif stop < 0:
            stop += self.shape[dim]
        if step is None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)

    def __getitem__(self, idxs):
        """
        Parameters
        ----------
        idxs: int | slice | tuple
            Indices to the subset of the n-dimensional array.

        Returns
        -------
        NDArray
            NDArray corresponding to the selected subset of elements.

        Raises
        ------
        AssertionError
            If a slice has negative step, or if number of slices is not equal
            to the number of dimensions.
        """
        idxs = tuplify(idxs)
        assert (
            len(idxs) == self.ndim
        ), "Need indexes equal to number of dimensions"
        idxs = tuple(
            (
                self._process_slice(s, i)
                if isinstance(s, slice)
                else slice(s, s + 1, 1)
            )
            for i, s in enumerate(idxs)
        )
        shape = tuple(int((s.stop - s.start - 1) / s.step) + 1 for s in idxs)
        strides = tuple(
            idx.step * stride for (idx, stride) in zip(idxs, self._strides)
        )
        return self.make(
            shape,
            strides,
            self._device,
            self._handle,
            sum(s.start * self._strides[i] for i, s in enumerate(idxs)),
        )

    def __setitem__(self, idxs, other):
        """
        Set the values of a view into an array, using the same semantics as
        __getitem__().
        """
        view = self.__getitem__(idxs)
        if isinstance(other, NDArray):
            assert prod(view.shape) == prod(other.shape)
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
        else:
            self.device.scalar_setitem(
                prod(view.shape),
                other,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )

    def ewise_or_scalar(self, other, ewise_func, scalar_func):
        """
        Run either an element-wise or scalar version of a function,
        depending on whether "other" is an NDArray or scalar
        """
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            assert (
                self.shape == other.shape
            ), "operation needs two equal-sized arrays"
            ewise_func(
                self.compact()._handle, other.compact()._handle, out._handle
            )
        else:
            scalar_func(self.compact()._handle, other, out._handle)
        return out

    def __add__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        return self * (-1)

    def __mul__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __pow__(self, other):
        assert isinstance(other, Number) and not isinstance(
            other, bool
        ), "array's power must be scalar"
        out = NDArray.make(self.shape, device=self.device)
        self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    def minimum(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_minimum, self.device.scalar_minimum
        )

    def __eq__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_eq, self.device.scalar_eq
        )

    def __ge__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_ge, self.device.scalar_ge
        )

    def __ne__(self, other):
        return 1 - (self == other)

    def __gt__(self, other):
        return (self >= other) * (self != other)

    def __lt__(self, other):
        return 1 - (self >= other)

    def __le__(self, other):
        return 1 - (self > other)

    def log(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out


# Convenience methods to match numpy a bit more closely.
def array(a, dtype="float32", device=None):
    assert dtype == "float32"
    return NDArray(a, device=device)


def empty(shape, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.empty(shape, dtype)


def full(shape, fill_value, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.full(shape, fill_value, dtype)


def broadcast_to(array, new_shape):
    return array.broadcast_to(new_shape)
