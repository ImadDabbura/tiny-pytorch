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
        return out_grad


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
        raise NotImplementedError()
        # return (
        #     out_grad
        #     * out_node.inputs[1]
        #     * out_node.inputs[0] ** (out_node.inputs[1] - 1)
        # )


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
        return Reshape(out_node.inputs[0].shape)(out_grad)


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


class Transpose(Op):
    def __init__(self, axes: tuple | None = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            self.axes = list(range(len(a.shape)))[-2:]
        return array_api.swapaxes(a, *self.axes)

    def gradient(self, out_grad, node):
        return Transpose(self.axes[::-1])(out_grad)


class MatMul(Op):
    def compute(self, x: NDArray, y: NDArray):
        return x @ y

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        x, y = out_node.inputs
        lhs = out_grad @ Transpose()(y)
        rhs = Transpose()(x) @ out_grad
        if lhs.shape != x.shape:
            lhs_sum_axis = len(lhs.shape) - len(x.shape)
            lhs = Summation(axes=tuple(range(lhs_sum_axis)))(lhs)
        if rhs.shape != y.shape:
            rhs_sum_axis = len(rhs.shape) - len(y.shape)
            rhs = Summation(axes=tuple(range(rhs_sum_axis)))(rhs)
        return lhs, rhs


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


class LogSumExp(Op):
    def __init__(self, axes: tuple | None = None):
        self.axes = axes

    def compute(self, Z):
        self.max = array_api.max(Z, axis=self.axes)
        if self.axes is None:
            axes = tuple(range(len(Z.shape)))
            self.max = array_api.array([self.max], dtype=Z.dtype)
        else:
            axes = self.axes
        self.tmp_shape = [1 if i in axes else x for i, x in enumerate(Z.shape)]
        tmp_max = array_api.reshape(self.max, tuple(self.tmp_shape))
        self.broadcasted_max = array_api.broadcast_to(tmp_max, Z.shape)
        return (
            array_api.log(
                array_api.sum(
                    array_api.exp(Z - self.broadcasted_max), self.axes
                )
            )
            + self.max
        )

    def gradient(self, out_grad, node):
        Z = node.inputs[0] - Tensor(self.broadcasted_max)
        log_sum_exp = BroadcastTo(Z.shape)(
            Reshape(tuple(self.tmp_shape))(Tensor(LogSumExp(self.axes)(Z)))
        )
        log_softmax = Z - log_sum_exp
        return BroadcastTo(Z.shape)(
            Reshape(tuple(self.tmp_shape))(out_grad)
        ) * (Exp()(log_softmax))
