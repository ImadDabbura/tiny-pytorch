"""Optimization module"""

import numpy as np

from . import init
from .tensor import Tensor


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue
            self.u[param] = (
                self.momentum
                * self.u.setdefault(param, init.zeros(*param.grad.shape))
                + (1 - self.momentum) * param.grad.data
            )
            param.data = Tensor(
                param.data * (1 - self.lr * self.weight_decay)
                - self.lr * self.u[param],
                dtype=param.dtype,
            )


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for param in self.params:
            if param.grad is None:
                continue
            grad = param.grad.data + self.weight_decay * param.data
            avg_grad = (
                self.beta1
                * self.m.setdefault(param, init.zeros(*param.grad.shape))
                + (1 - self.beta1) * grad
            )
            self.m[param] = avg_grad
            avg_grad /= 1 - self.beta1**self.t
            sqr_grad = self.beta2 * self.v.setdefault(
                param, init.zeros(*param.grad.shape)
            ) + (1 - self.beta2) * (grad**2)
            self.v[param] = sqr_grad
            sqr_grad /= 1 - self.beta2**self.t
            param.data = Tensor(
                param.data - (self.lr * avg_grad / (sqr_grad**0.5 + self.eps)),
                dtype=param.dtype,
            )
