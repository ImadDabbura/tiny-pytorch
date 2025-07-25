"""NDArray implementation with multiple backend support.

This module provides the core NDArray class that supports multiple backends
including NumPy, CPU, and CUDA. The NDArray class is a Python wrapper for
handling operations on n-dimensional arrays with strided memory layout.

The module includes:
- BackendDevice class for device abstraction
- NDArray class with strided array operations
- Device factory functions (cpu, cuda, etc.)
- Array creation utilities

Key Features
-----------
- Strided memory layout for efficient array operations
- Multiple backend support (NumPy, CPU, CUDA)
- Broadcasting and reshaping without memory copying
- Element-wise and scalar operations
- Matrix operations and reductions

Classes
-------
BackendDevice
    Device abstraction that wraps backend implementation modules.
NDArray
    Multi-dimensional array with strided memory layout.

Functions
---------
cpu_numpy, cpu, cuda
    Device factory functions.
array, empty, full
    Array creation utilities.
broadcast_to
    Broadcasting utility function.
reshape
    Reshape utility function.
maximum
    Element-wise maximum function.
log
    Natural logarithm function.
exp
    Exponential function.
tanh
    Hyperbolic tangent function.
summation
    Sum of array elements over a given axis.
"""

from math import prod
from numbers import Number

import numpy as np

from ..utils import tuplify
from . import ndarray_backend_numpy


class BackendDevice:
    """Backend device that wraps the implementation module for each device.

    This class provides a unified interface for different backend implementations
    (numpy, CPU, CUDA) by forwarding operations to the appropriate module.

    Attributes
    ----------
    name : str
        Name of the device (e.g., "cpu", "cuda", "cpu_numpy").
    module : object
        The backend implementation module that handles actual operations.
    """

    def __init__(self, name: str, module=None):
        """Initialize a new BackendDevice.

        Parameters
        ----------
        name : str
            Name of the device.
        module : object, optional
            Module that implements the device's operations.
        """
        self.name = name
        # Key attribute that will handle all ops on its device
        self.module = module

    def __eq__(self, other):
        """Check if two devices are equal.

        Two devices are equal if they have the same name.

        Parameters
        ----------
        other : object
            Device to compare with.

        Returns
        -------
        bool
            True if devices have the same name.
        """
        return self.name == other.name

    def __repr__(self):
        """String representation of the device.

        Returns
        -------
        str
            String representation showing the device name.
        """
        return f"{self.name}()"

    def enabled(self):
        """Check if the device is enabled.

        Returns
        -------
        bool
            True if the device has an implementation module.
        """
        return self.module is not None

    def __getattr__(self, name):
        """Forward attribute access to the implementation module.

        All attempts to get attribute from device will be forwarded to the
        module that implements the device's operations.
        i.e. device.op will become self.module.op

        Parameters
        ----------
        name : str
            Name of the attribute to access.

        Returns
        -------
        object
            Attribute from the implementation module.
        """
        return getattr(self.module, name)

    def randn(self, *shape, dtype="float32"):
        """Generate random numbers from standard normal distribution.

        Parameters
        ----------
        *shape : int
            Shape of the output array.
        dtype : str, optional
            Data type of the array. Default is "float32".

        Returns
        -------
        NDArray
            Array with random values from N(0, 1).
        """
        return NDArray(np.random.randn(*shape).astype(dtype), device=self)

    def rand(self, *shape, dtype="float32"):
        """Generate random numbers from uniform distribution.

        Parameters
        ----------
        *shape : int
            Shape of the output array.
        dtype : str, optional
            Data type of the array. Default is "float32".

        Returns
        -------
        NDArray
            Array with random values from U[0, 1).
        """
        return NDArray(np.random.rand(*shape).astype(dtype), device=self)

    def one_hot(self, n, i, dtype="float32"):
        """Create a one-hot encoded array.

        Parameters
        ----------
        n : int
            Number of classes.
        i : int or array-like
            Indices to encode.
        dtype : str, optional
            Data type of the array. Default is "float32".

        Returns
        -------
        NDArray
            One-hot encoded array.
        """
        return NDArray(np.eye(n, dtype=dtype)[i], device=self)

    def empty(self, shape, dtype="float32"):
        """Create an empty array.

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the array.
        dtype : str, optional
            Data type of the array. Default is "float32".

        Returns
        -------
        NDArray
            Empty array with the specified shape.
        """
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return NDArray.make(shape, device=self)

    def full(self, shape, fill_value, dtype="float32"):
        """Create an array filled with a constant value.

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the array.
        fill_value : float
            Value to fill the array with.
        dtype : str, optional
            Data type of the array. Default is "float32".

        Returns
        -------
        NDArray
            Array filled with the specified value.
        """
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr


def cpu_numpy():
    """Create a CPU device using NumPy backend.

    Returns
    -------
    BackendDevice
        CPU device with NumPy backend.
    """
    return BackendDevice("cpu_numpy", ndarray_backend_numpy)


def default_device():
    """Return the default device (CPU with NumPy backend).

    Returns
    -------
    BackendDevice
        Default CPU device.
    """
    return cpu_numpy()


def cpu():
    """Create a CPU device with native backend if available.

    Attempts to use the native CPU backend, falls back to NumPy if
    the C++ extension is not available.

    Returns
    -------
    BackendDevice
        CPU device with best available backend.
    """
    try:
        from .. import ndarray_backend_cpu

        return BackendDevice("cpu", ndarray_backend_cpu)
    except ImportError:
        # Fallback to numpy if C++ extension is not available
        return cpu_numpy()


def cuda():
    """Create a CUDA device if available.

    Returns
    -------
    BackendDevice
        CUDA device, or disabled device if CUDA is not available.
    """
    try:
        from .. import ndarray_backend_cuda

        return BackendDevice("cuda", ndarray_backend_cuda)
    except ImportError:
        return BackendDevice("cuda", None)


def all_devices():
    """Get a list of all available devices.

    Returns
    -------
    list[BackendDevice]
        List of all available devices.
    """
    return [cpu_numpy(), cpu(), cuda()]


class NDArray:
    """A generic ND array class that may contain multiple different backends.

    NDArray is basically a Python wrapper for handling operations on
    n-dimensional arrays. The underlying array is just a flat 1D array and the
    backend device will handle all the ops on the 1D array. Strides, shape, and
    offset allows us to map n-dimensional (logical) array to the 1D flat array
    that are physically allocated in memory.

    The high level ops such as broadcasting and transposing are all done in
    Python without touching the underlying array. The other `raw` ops such as
    addition and matrix-multiplication will be implemented in C/C++ that would
    call highly optimized Kernels for such ops such as CUDA Kernels.

    To make the backend code simpler, we will only operate on compact arrays so
    we need to call `compact()` before any ops AND we support only float32 data
    type.

    Attributes
    ----------
    device : BackendDevice
        Device that handles the operations.
    shape : tuple[int, ...]
        Shape of the array.
    strides : tuple[int, ...]
        Strides for accessing elements in the underlying 1D array.
    size : int
        Total number of elements in the array.
    ndim : int
        Number of dimensions in the array.
    dtype : str
        Data type of the array (currently only "float32" is supported).
    """

    def __init__(self, other, device=None):
        """Create NDArray by copying another NDArray/Numpy array OR use Numpy as
        a bridge for all other types of iterables.

        Parameters
        ----------
        other : NDArray or numpy.ndarray or array_like
            Source data to create the NDArray from.
        device : BackendDevice, optional
            Device to place the array on. If None, uses default device.
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
        """Initialize the NDArray from another NDArray.

        Parameters
        ----------
        other : NDArray
            Source NDArray to copy attributes from.
        """
        # shape, stride, and offset allows us to represent 1D array as NDarray
        self._shape = other.shape
        self._strides = other.strides
        self._offset = other._offset
        # BackendDevice: helps us dispatch ops to the corresponding device
        self._device = other.device
        self._handle = other._handle  # pointer to 1D array of type `Array`

    def to(self, device):
        """Move the array to a different device.

        Parameters
        ----------
        device : BackendDevice
            Target device.

        Returns
        -------
        NDArray
            Array on the target device.
        """
        if device == self.device:
            return self
        # Use Numpy as a bridge by first converting NDArray to numpy array first
        return NDArray(self.numpy(), device)

    def numpy(self):
        """Convert the NDArray to a NumPy array.

        Returns
        -------
        numpy.ndarray
            NumPy array with the same data.
        """
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
        return prod(self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        # Only supporting float32 for now
        return "float32"

    @property
    def flat(self):
        """Return a 1-D view (flattened) of the array.

        Returns
        -------
        NDArray
            A 1-dimensional view of the array with the same data.

        Examples
        --------
        >>> a = NDArray([[1, 2], [3, 4]])
        >>> a.flat
        NDArray([1, 2, 3, 4], device=cpu_numpy())
        """
        return self.reshape((self.size,))

    def __str__(self):
        return str(self.numpy())

    def __repr__(self):
        return f"NDArray({str(self)}, device={self.device})"

    def fill(self, value):
        """Fill in-place with a constant value."""
        self.device.fill(self._handle, value)

    def is_compact(self):
        """
        Return true if array is compact in memory and internal size equals
        product of the shape dimensions.
        """
        # An array is contiguous is it has the same strides as row-major order
        # and offset = 0 (i.e. same size as the original array)
        return (
            self.strides == NDArray.compact_strides(self.shape)
            and self._handle.size == self.size
        )

    def compact(self):
        """Convert NDArray to be compact if it is not already compact."""
        if self.is_compact():
            return self
        out = NDArray.make(self.shape, device=self.device)
        self.device.compact(
            self._handle, out._handle, self.shape, self.strides, self._offset
        )
        return out

    def as_strided(self, shape, strides):
        """
        Create a strided view of the underlying memory without copying anything.
        """
        assert len(shape) == len(strides)
        return NDArray.make(shape, strides, self.device, self._handle)

    def reshape(self, new_shape):
        """
        Reshape the array without copying memory.  This will return a new array
        (view) that corresponds to a reshaped array but points to the same
        memory as the original array. Therefore, we only change the shape and
        the strides to get the new n-dimensional logical view of the array.

        Parameters
        ----------
        new_shape: tuple
            New shape of the array.

        Returns
        -------
        NDArray
            Reshaped array; this will point to the same memory as the original
            NDArray.

        Raises
        ------
        ValueError
            If product of current shape is not equal to the product of the new
            shape, or if the matrix is not compact.
        """
        if prod(self.shape) != prod(new_shape) or not self.is_compact():
            raise ValueError(
                "Array must be compact and its size must remain the same."
            )
        new_strides = self.compact_strides(new_shape)
        return self.make(
            new_shape, new_strides, self.device, self._handle, self._offset
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
            Permutation order of the dimensions.

        Returns
        -------
        NDarray
            New NDArray object with permuted dimensions, pointing to the same
            memory as the original NDArray (i.e., just shape and strides changed).
        """
        assert len(self.shape) == len(new_axes)
        shape = tuple(self.shape[i] for i in new_axes)
        strides = tuple(self.strides[i] for i in new_axes)
        return self.make(
            shape, strides, self.device, self._handle, self._offset
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
            Shape to broadcast to.

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
        assert len(self.shape) == len(new_shape)
        assert all(
            e == new_shape[i] for i, e in enumerate(self.shape) if e != 1
        )
        new_strides = tuple(
            0 if e == 1 else self.strides[i] for i, e in enumerate(self.shape)
        )
        return self.make(
            new_shape, new_strides, self.device, self._handle, self._offset
        )

    def _process_idx(self, sl: int | slice, dim: int):
        """Convert index/slice to indices within bound."""
        if isinstance(sl, slice):
            return slice(*sl.indices(self.shape[dim]))
        if abs(sl) > self.shape[dim]:
            raise IndexError(
                f"index {sl} is out of bounds for axis {dim} with size {self.shape[dim]}"
            )
        return slice(*slice(sl, sl + 1, 1).indices(self.shape[dim]))

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
        TypeError
            If an index is not an int or slice.
        """
        idxs = tuplify(idxs)
        assert (
            len(idxs) == self.ndim
        ), "Need indexes equal to number of dimensions"
        # Type check for each index
        for s in idxs:
            if not isinstance(s, (int, slice)):
                raise TypeError(
                    f"Invalid index type: {type(s)}. Only int or slice allowed."
                )
        idxs = tuple(self._process_idx(s, i) for i, s in enumerate(idxs))
        shape = tuple(int((s.stop - s.start - 1) / s.step) + 1 for s in idxs)
        strides = tuple(
            idx.step * stride for (idx, stride) in zip(idxs, self.strides)
        )
        return self.make(
            shape,
            strides,
            self.device,
            self._handle,
            sum(s.start * self.strides[i] for i, s in enumerate(idxs)),
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
        depending on whether "other" is an NDArray or scalar.
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

    def __matmul__(self, other):
        """
        Matrix multiplication of two arrays.  This requires that both arrays
        be 2D (i.e., we don't handle batch matrix multiplication), and that the
        sizes match up properly for matrix multiplication.
        """
        assert self.ndim == 2 and other.ndim == 2
        assert self.shape[1] == other.shape[0]

        m, n, p = self.shape[0], self.shape[1], other.shape[1]

        # if the matrix is aligned, use tiled matrix multiplication
        if hasattr(self.device, "matmul_tiled") and all(
            d % self.device.__tile_size__ == 0 for d in (m, n, p)
        ):

            def tile(a, tile):
                return a.as_strided(
                    (m // tile, n // tile, tile, tile),
                    (n * tile, tile, n, 1),
                )

            t = self.device.__tile_size__
            a = tile(self.compact(), t).compact()
            b = tile(other.compact(), t).compact()
            out = NDArray.make((m // t, p // t, t, t), device=self.device)
            self.device.matmul_tiled(
                a._handle, b._handle, out._handle, m, n, p
            )

            return out.permute((0, 2, 1, 3)).compact().reshape((m, p))

        else:
            out = NDArray.make((m, p), device=self.device)
            self.device.matmul(
                self.compact()._handle,
                other.compact()._handle,
                out._handle,
                m,
                n,
                p,
            )
            return out

    def reduce_view_out(self, axis, keepdims=False):
        """
        Return a view to the array set up for reduction functions and output
        array.
        """
        if isinstance(axis, tuple) and not axis:
            raise ValueError("Empty axis in reduce")

        if axis is None:
            view = self.compact().reshape(
                (1,) * (self.ndim - 1) + (prod(self.shape),)
            )
            out = NDArray.make(
                (1,) * (self.ndim if keepdims else 1), device=self.device
            )

        else:
            if isinstance(axis, (tuple, list)):
                assert (
                    len(axis) == 1
                ), "Only support reduction over a single axis"
                axis = axis[0]

            if abs(axis) > self.ndim:
                raise ValueError(
                    f"Dimension out of range (expected to be in range of [-{self.ndim}, {self.ndim - 1}], but got {axis})"
                )
            axis = axis + self.ndim if axis < 0 else axis
            print(axis)
            view = self.permute(
                tuple([a for a in range(self.ndim) if a != axis]) + (axis,)
            )
            out = NDArray.make(
                (
                    tuple(
                        [
                            1 if i == axis else s
                            for i, s in enumerate(self.shape)
                        ]
                    )
                    if keepdims
                    else tuple(
                        [s for i, s in enumerate(self.shape) if i != axis]
                    )
                ),
                device=self.device,
            )
        return view, out

    def sum(self, axis=None, keepdims=False):
        """
        Sum either across all axis (when axis=None) or one axis.

        Note: It doesn't support axis being multiple of axes.
        """
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_sum(
            view.compact()._handle, out._handle, view.shape[-1]
        )
        return out

    def max(self, axis=None, keepdims=False):
        """
        Max either across all axis (when axis=None) or one axis.

        Note: It doesn't support axis being multiple of axes.
        """
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_max(
            view.compact()._handle, out._handle, view.shape[-1]
        )
        return out

    def pad(self, axes: tuple[tuple[int, int], ...]) -> "NDArray":
        """
        Pad the array with zeros along each axis according to the specified padding widths.

        Parameters
        ----------
        axes : tuple[tuple[int, int], ...]
            A tuple specifying the number of values padded to the edges of each axis.
            Each element should be a tuple of two integers (pad_before, pad_after),
            where pad_before is the number of values padded before the first element
            and pad_after is the number of values padded after the last element for that axis.
            The length of axes must match the number of dimensions of the array.

        Returns
        -------
        NDArray
            A new NDArray with the specified padding applied, filled with zeros in the padded regions.

        Raises
        ------
        AssertionError
            If the length of axes does not match the number of dimensions of the array.

        Examples
        --------
        >>> a = NDArray([[1, 2], [3, 4]])
        >>> a.pad(((1, 1), (2, 2)))
        NDArray(
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 1, 2, 0, 0],
             [0, 0, 3, 4, 0, 0],
             [0, 0, 0, 0, 0, 0]], device=cpu_numpy())
        """
        assert len(axes) == len(
            self.shape
        ), "Each dimension should have its own tuple of left and right padding"
        new_shape = tuple([l + r + n for (l, r), n in zip(axes, self.shape)])
        arr = self.device.full(new_shape, 0)
        old_idxs = tuple(
            [slice(l, n - r) for (l, r), n in zip(axes, new_shape)]
        )
        arr[old_idxs] = self
        return arr


# Convenience methods to match numpy a bit more closely.
def array(a, dtype="float32", device=None):
    """Create an NDArray from array-like data.

    Parameters
    ----------
    a : array_like
        Input data to create the array from.
    dtype : str, optional
        Data type of the array. Default is "float32".
    device : BackendDevice, optional
        Device to place the array on. If None, uses default device.

    Returns
    -------
    NDArray
        New NDArray with the specified data.
    """
    dtype = "float32" if dtype is None else dtype
    assert dtype == "float32"
    return NDArray(a, device=device)


def empty(shape, dtype="float32", device=None):
    """Create an empty NDArray.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the array.
    dtype : str, optional
        Data type of the array. Default is "float32".
    device : BackendDevice, optional
        Device to place the array on. If None, uses default device.

    Returns
    -------
    NDArray
        Empty NDArray with the specified shape.
    """
    return NDArray.make(shape, device=device)


def full(shape, fill_value, dtype="float32", device=None):
    """Create an NDArray filled with a constant value.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the array.
    fill_value : float
        Value to fill the array with.
    dtype : str, optional
        Data type of the array. Default is "float32".
    device : BackendDevice, optional
        Device to place the array on. If None, uses default device.

    Returns
    -------
    NDArray
        NDArray filled with the specified value.
    """
    arr = empty(shape, dtype, device)
    arr.fill(fill_value)
    return arr


def broadcast_to(array, new_shape):
    """Broadcast an array to a new shape.

    Parameters
    ----------
    array : NDArray
        Array to broadcast.
    new_shape : tuple[int, ...]
        Target shape for broadcasting.

    Returns
    -------
    NDArray
        Broadcasted array.
    """
    return array.broadcast_to(new_shape)


def reshape(array: NDArray, new_shape: tuple[int, ...]) -> NDArray:
    """Reshape an array to a new shape.

    Parameters
    ----------
    array : NDArray
        Array to reshape.
    new_shape : tuple[int, ...]
        New shape of the array.

    Returns
    -------
    NDArray
        Reshaped array.

    Raises
    ------
    ValueError
        If the product of the new shape is not equal to the product of the
        original shape, or if the array is not compact.
    """
    return array.reshape(new_shape)


def maximum(a: NDArray, b: NDArray | float) -> NDArray:
    """Element-wise maximum of array elements.

    Parameters
    ----------
    a : NDArray
        First array.
    b : NDArray or float
        Second array or scalar value.

    Returns
    -------
    NDArray
        Array containing the element-wise maximum of a and b.
    """
    return a.maximum(b)


def log(a: NDArray) -> NDArray:
    """Natural logarithm, element-wise.

    Parameters
    ----------
    a : NDArray
        Input array.

    Returns
    -------
    NDArray
        Natural logarithm of a, element-wise.
    """
    return a.log()


def exp(a: NDArray) -> NDArray:
    """Exponential, element-wise.

    Parameters
    ----------
    a : NDArray
        Input array.

    Returns
    -------
    NDArray
        Exponential of a, element-wise.
    """
    return a.exp()


def tanh(a: NDArray) -> NDArray:
    """Hyperbolic tangent, element-wise.

    Parameters
    ----------
    a : NDArray
        Input array.

    Returns
    -------
    NDArray
        Hyperbolic tangent of a, element-wise.
    """
    return a.tanh()


def summation(
    a: NDArray, axis: int | None = None, keepdims: bool = False
) -> NDArray:
    """Sum of array elements over a given axis.

    Parameters
    ----------
    a : NDArray
        Input array.
    axis : int or None, optional
        Axis along which a sum is performed. The default, axis=None,
        will sum all of the elements of the input array. If axis is negative
        it counts from the last to the first axis.

        Note: Only supports reduction over a single axis or all axes.
        Multiple axes reduction is not supported.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result will
        broadcast correctly against the input array.

    Returns
    -------
    NDArray
        An array with the same shape as a, with the specified axis removed.
        If a is a 0-d array, or if axis is None, a scalar is returned.
        If an output array is specified, a reference to out is returned.

    Raises
    ------
    ValueError
        If an empty axis tuple is provided.
    """
    return a.sum(axis=axis, keepdims=keepdims)


def negative(a: NDArray) -> NDArray:
    """Numerical negative, element-wise.

    Parameters
    ----------
    a : NDArray
        Input array.

    Returns
    -------
    NDArray
        Returned array or scalar: y = -x.

    Examples
    --------
    >>> a = NDArray([1, -1, 2.5])
    >>> negative(a)
    NDArray([-1.0, 1.0, -2.5], device=cpu_numpy())

    >>> b = NDArray([[1, 2], [3, 4]])
    >>> negative(b)
    NDArray([[-1.0, -2.0], [-3.0, -4.0]], device=cpu_numpy())
    """
    return -a
