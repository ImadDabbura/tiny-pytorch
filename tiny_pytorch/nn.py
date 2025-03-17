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


# TODO: Both weight and biases are initialized with Kaiming uniform
class Linear(Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            init.kaiming_uniform(
                in_features,
                out_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        if bias:
            # TODO: bias should be 1D vector and initialized to zero
            self.bias = Parameter(
                ops.Reshape((1, out_features))(
                    init.kaiming_uniform(
                        out_features,
                        1,
                        device=device,
                        dtype=dtype,
                        requires_grad=True,
                    )
                )
            )

    def forward(self, X: Tensor) -> Tensor:
        out = X @ self.weight
        if self.bias:
            bias = ops.Reshape(
                ((1,) * len(X.shape[:-1])) + (self.out_features,)
            )(self.bias)
            bias = ops.BroadcastTo(tuple(X.shape[:-1]) + (self.out_features,))(
                bias
            )
            out += bias
        return out
