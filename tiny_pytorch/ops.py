from __future__ import annotations

from itertools import zip_longest

import numpy as array_api

from .tensor import Op, Tensor

NDArray = array_api.ndarray


class ScalarAdd(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, x: NDArray):
        return self.scalar + x

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return out_grad * self.scalar


class EWiseAdd(Op):
    def compute(self, x: NDArray, y: NDArray):
        return x + y

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return out_grad, out_grad


class ScalarMul(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, x: NDArray):
        return self.scalar * x

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return out_grad * self.scalar


class EWiseMul(Op):
    def compute(self, x: NDArray, y: NDArray):
        return x * y

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return out_grad * out_node.inputs[1], out_grad * out_node.inputs[0]


class Negate(Op):
    def compute(self, x: NDArray):
        return array_api.negative(x)

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return -out_grad


class ScalarPower(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, x: NDArray):
        return x**self.scalar

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return out_grad * self.scalar * out_node.inputs[0] ** (self.scalar - 1)


class EWisePower(Op):
    def compute(self, x: NDArray, y: NDArray):
        return x**y

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return (
            out_grad
            * out_node.inputs[1]
            * out_node.inputs[0] ** (out_node.inputs[1] - 1)
        )


class ScalarDivide(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, x: NDArray):
        return x / self.scalar

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return out_grad / self.scalar


class EWiseDivide(Op):
    def compute(self, x: NDArray, y: NDArray):
        return x / y

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return out_grad / out_node.inputs[1], out_grad * (
            -out_node.inputs[0] / out_node.inputs[1] ** 2
        )


class Reshape(Op):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, x: NDArray):
        return array_api.reshape(x, self.shape)

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return array_api.reshape(out_grad, out_node.inputs[0])


class Summation(Op):
    def __init__(self, axes: tuple | None = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, self.axes)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        axes = self.axes if self.axes else tuple(range(len(input_shape)))
        tmp_shape = [1 if i in axes else x for i, x in enumerate(input_shape)]
        tmp_out = Reshape(tuple(tmp_shape))(out_grad)
        return BroadcastTo(input_shape)(tmp_out)


class BroadcastTo(Op):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        axes = [
            i
            for i, x in enumerate(zip_longest(input_shape, out_grad.shape))
            if x[0] == 1 or not x[0]
        ]
        out = Summation(tuple(axes))(out_grad)
        out = Reshape(input_shape)(out)
        return out


class Log(Op):
    def compute(self, x: NDArray):
        return array_api.log(x)

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return out_grad / out_node.inputs[0]


class Exp(Op):
    def compute(self, x: NDArray):
        return array_api.exp(x)

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return out_grad * Exp()(out_node.inputs[0])


class ReLU(Op):
    def compute(self, x: NDArray):
        return array_api.maximum(0, x)

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return out_grad * (out_node.realize_cached_data() > 0)
