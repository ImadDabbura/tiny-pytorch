from functools import reduce
from operator import mul
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


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.ReLU()(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        m, k = logits.shape
        y_one_hot = init.one_hot(k, y.numpy().tolist())
        log_sum_exp = ops.BroadcastTo((m, k))(
            ops.Reshape((m, 1))(ops.LogSumExp((1,))(logits))
        )
        log_softmax = logits - log_sum_exp
        return -ops.Summation()(log_softmax * y_one_hot) / m


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        m = x.shape[0]
        mean = ops.Summation((1,))(x) / self.dim
        mean = ops.BroadcastTo((m, self.dim))(ops.Reshape((m, 1))(mean))
        var = ops.Summation((1,))((x - mean) ** 2) / self.dim
        var = ops.BroadcastTo((m, self.dim))(ops.Reshape((m, 1))(var))
        x = (x - mean) / ((var + self.eps) ** 0.5)
        weight = ops.BroadcastTo((m, self.dim))(
            ops.Reshape((1, self.dim))(self.weight)
        )
        bias = ops.BroadcastTo((m, self.dim))(
            ops.Reshape((1, self.dim))(self.bias)
        )
        if self.training:
            return weight * x + bias
        return weight.data * x + bias.data


class Flatten(Module):
    def forward(self, X):
        shape = X.shape
        return ops.Reshape((shape[0], reduce(mul, shape[1:])))(X)
