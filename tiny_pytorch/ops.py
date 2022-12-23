from __future__ import annotations

import numpy as array_api

from . import tensor

NDArray = array_api.ndarray


class Op:
    def __call__(self, *args):
        return tensor.Tensor.from_operation(self, args)

    def compute(self, *args: tuple[NDArray]):
        raise NotImplementedError()

    def gradient(self, out_grad, out_node):
        raise NotImplementedError()


class ScalarAdd(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, x: NDArray):
        return self.scalar + x

    def gradient(self, out_grad: tensor.Tensor, out_node: tensor.Tensor):
        return out_grad * self.scalar


class EWiseAdd(Op):
    def compute(self, x: NDArray, y: NDArray):
        return x + y

    def gradient(self, out_grad: tensor.Tensor, out_node: tensor.Tensor):
        return out_grad, out_grad


class ScalarMul(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, x: NDArray):
        return self.scalar * x

    def gradient(self, out_grad: tensor.Tensor, out_node: tensor.Tensor):
        return out_grad * self.scalar


class EWiseMul(Op):
    def compute(self, x: NDArray, y: NDArray):
        return x * y

    def gradient(self, out_grad: tensor.Tensor, out_node: tensor.Tensor):
        return out_grad * out_node.inputs[1], out_grad * out_node.inputs[0]


class Negate(Op):
    def compute(self, x: NDArray):
        return array_api.negative(x)

    def gradient(self, out_grad: tensor.Tensor, out_node: tensor.Tensor):
        return -out_grad
