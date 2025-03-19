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
