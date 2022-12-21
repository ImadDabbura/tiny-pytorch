r"""
Core data structures for multi-dimensional tensors.
"""
from __future__ import annotations

import numpy as np
import numpy as array_api

from .device import CPUDevice, Device, cpu
from .ops import Op
from .utils import listify

NDArray = array_api.ndarray


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
            device = array.device if device else device
            dtype = array.dtype if dtype else dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # Use numpy as brige
                cached_data = Tensor._from_numpy_array(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = cpu() if device else device
            cached_data = self._from_numpy_array(
                array, device=device, dtype=dtype
            )
        self._init(cached_data=cached_data, requires_grad=requires_grad)

    def _init(
        self,
        inputs: list[Tensor] | None = None,
        op: Op | None = None,
        *,
        cached_data: list[object],
        requires_grad: bool | None = None,
    ):
        if requires_grad is None:
            # If any of the input Tensors have requires_grad -> output will requires_grad
            requires_grad = any([x.requires_grad for x in inputs])
        self.inputs = listify(inputs)
        self.op = op
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    def _from_numpy_array(self, array, device, dtype):
        if array_api is np:
            return np.array(array, dtype=dtype)
        return array_api.array(array, device=device, dtype=dtype)

    def realize_cached_data(self):
        """Run computation to get the output if the LAZY MODE is on, else return cached data."""
        if self.cached_data is not None:
            return self.cached_data
        self.cached_data = self.op.compute(
            [x.realize_cached_data() for x in self.inputs]
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
        return self.realize_cached_data().shape

    @property
    def ndim(self):
        return self.realize_cached_data().ndim

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        if array_api is np:
            return cpu()
        return self.realize_cached_data().device

    @property
    def data(self):
        """Returns a detached Tensor with the original data."""
        return self.detach()

    @data.setter
    def data(self, data):
        assert isinstance(data, Tensor), "data must be of type `Tensor`"
        assert self.dtype == data.dtype, "data must be of the the same type"
        self.cached_data = data.realize_cached_data()

    @staticmethod
    def from_constant(data, requires_grad: bool = False):
        """Creates a leaf node Tensor from the given `data`."""
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            cached_data=data.realize_cached_data(), requires_grad=False
        )
        return tensor

    def detach(self):
        """
        Returns a new Tensor with no history (detached from the computation
        graph). The returned Tensor will share the same data with the
        original one.
        """
        return Tensor.from_constant(self)
