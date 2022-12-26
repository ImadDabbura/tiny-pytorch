from __future__ import annotations

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
