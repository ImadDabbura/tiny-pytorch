r"""
Core data structures for multi-dimensional tensors.
"""
from __future__ import annotations

import numpy as np
import numpy as array_api

import tiny_pytorch

from .device import CPUDevice, Device, cpu
from .utils import listify

NDArray = array_api.ndarray
LAZY_MODE = False  # Default mode is eager mode


class Op:
    def __call__(self, *args):
        return Tensor.from_operation(self, args)

    def compute(self, *args: tuple[NDArray]):
        raise NotImplementedError()

    def gradient(self, out_grad, out_node):
        raise NotImplementedError()


class Tensor:
    def __init__(
        self,
        array,
        *,
        device: Device | None = None,
        dtype: str | None = None,
        requires_grad: bool = False,
    ) -> None:
        """Construct a Tensor with no autograd history by copying `array`."""
        if isinstance(array, Tensor):
            device = array.device if not device else device
            dtype = array.dtype if not dtype else dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array._realize_cached_data()
            else:
                # Use numpy as brige
                cached_data = self._from_numpy_array(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = cpu() if not device else device
            cached_data = self._from_numpy_array(
                array, device=device, dtype=dtype
            )
        self._init(cached_data=cached_data, requires_grad=requires_grad)

    def _init(
        self,
        inputs: list[Tensor] | None = None,
        op: Op | None = None,
        *,
        cached_data: list[object] | None = None,
        requires_grad: bool | None = None,
    ):
        self.inputs = listify(inputs)
        self.op = op
        self.cached_data = cached_data
        if requires_grad is None:
            # If any of the input Tensors have requires_grad -> output will requires_grad
            requires_grad = any([x.requires_grad for x in self.inputs])
        self.requires_grad = requires_grad

    def _from_numpy_array(self, array, device, dtype):
        if array_api is np:
            return np.array(array, dtype=dtype)
        return array_api.array(array, device=device, dtype=dtype)

    def _realize_cached_data(self):
        """Run computation to get the output if the LAZY MODE is on, else return cached data."""
        if self.cached_data is not None:
            return self.cached_data
        self.cached_data = self.op.compute(
            *[x._realize_cached_data() for x in self.inputs]
        )
        return self.cached_data

    def numpy(self):
        """
        Returns `Tensor` as Numpy ndarray. The underlying data will be shared
        between Tensor and the Numpy ndarray.
        """
        data = self._realize_cached_data()
        if array_api is np:
            return data
        return data.numpy()

    @property
    def shape(self):
        return self._realize_cached_data().shape

    @property
    def ndim(self):
        return self._realize_cached_data().ndim

    @property
    def dtype(self):
        return self._realize_cached_data().dtype

    @property
    def device(self):
        if array_api is np:
            return cpu()
        return self._realize_cached_data().device

    @property
    def data(self):
        """Returns a detached Tensor with the original data."""
        return self.detach()

    @data.setter
    def data(self, data):
        assert isinstance(
            data, Tensor
        ), f"data must be of type `Tensor`, {type(data)} is given"
        assert self.dtype == data.dtype, (
            f"data must be of the same type as `Tensor`, "
            f"{self.dtype} != {data.dtype}"
        )

    @staticmethod
    def from_constant(data, requires_grad: bool = False):
        """Creates a leaf node Tensor from the given `data`."""
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            cached_data=data._realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @staticmethod
    def from_operation(op: Op, inputs: tuple[Tensor]):
        """
        Creates a node Tensor by applying the `op` operation on the `inputs`
        Tensors.
        """
        tensor = Tensor.__new__(Tensor)
        tensor._init(inputs, op)
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            return tensor._realize_cached_data()
        return tensor

    def detach(self):
        """
        Returns a new Tensor with no history (detached from the computation
        graph). The returned Tensor will share the same data with the
        original one.
        """
        return Tensor.from_constant(self)

    def is_leaf(self):
        """
        All Tensors that have `requires_grad` set to `False` OR they were
        created by the user and were not the result of an operation are
        considered leaf Tensors.
        """
        return self.op is None

    def __repr__(self):
        return f"tiny_pytorch.Tensor({str(self._realize_cached_data())})"

    def __str__(self):
        return str(self._realize_cached_data())

    def __add__(self, other):
        if isinstance(other, Tensor):
            return tiny_pytorch.ops.EWiseAdd()(self, other)
        return tiny_pytorch.ops.ScalarAdd(other)(self)

    def __neg__(self):
        return tiny_pytorch.ops.Negate()(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return tiny_pytorch.ops.EWiseAdd()(self, -other)
        return tiny_pytorch.ops.ScalarAdd(-other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return tiny_pytorch.ops.EWiseMul()(self, other)
        return tiny_pytorch.ops.ScalarMul(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return tiny_pytorch.ops.EWisePower()(self, other)
        return tiny_pytorch.ops.ScalarPower(other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return tiny_pytorch.ops.EWiseDivide()(self, other)
        return tiny_pytorch.ops.ScalarDivide(other)(self)

    def __matmul__(self, other):
        return tiny_pytorch.ops.MatMul()(self, self, other)

    def sum(self, axes=None):
        return tiny_pytorch.ops.Summation(axes)(self)

    def reshape(self, shape):
        return tiny_pytorch.ops.Reshape(shape)(self)

    def broadcast_to(self, shape):
        return tiny_pytorch.ops.BroadcastTo(shape)(self)

    def transpose(self, axes=None):
        return tiny_pytorch.ops.Transpose(axes)(self)

    def backward(self, out_grad: Tensor | None = None):
        out_grad = out_grad if out_grad else self.device.ones(self.shape)
        self._compute_gradients(out_grad)

    def _compute_gradients(self, out_grad):
        pass

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rmatmul__ = __matmul__
