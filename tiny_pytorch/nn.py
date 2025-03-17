from typing import Any

from . import init, ops
from .tensor import Tensor


class Parameter(Tensor):
    """
    A special kind of tensor that represents parameters. It acts as a marker
    so modules can be able to identify learnable parameters. All `Parameter`
    tensors have require_grad set to True.
    """


class Module:
    def __init__(self):
        self._training = True
        self._params = []
        self._children = []

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, value):
        self._training = value
        for module in self._children:
            module.training = value

    def register_params(self, *params):
        self._params += params

    def register_modules(self, *modules):
        self._children += modules

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return self._params + sum(
            [module.parameters() for module in self._children], []
        )

    def children(self):
        return self._children

    def eval(self):
        self._training = False
        for module in self.children():
            module.training = False

    def train(self):
        self._training = True
        for module in self.children():
            module.training = True

    def __setattr__(self, k, v):
        super().__setattr__(k, v)
        if isinstance(v, Parameter):
            self.register_params(v)
        elif isinstance(v, Module):
            self.register_modules(v)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
