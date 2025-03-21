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


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for v in value.values():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self._training = True

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, value):
        self._training = value
        for module in self.children():
            module.training = value

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

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


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = init.randb(*x.shape, p=(1 - self.p))
            return (mask * x) / (1 - self.p)
        return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x


class BatchNorm1d(Module):
    def __init__(
        self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(
            dim, device=device, dtype=dtype, requires_grad=False
        )
        self.running_var = init.ones(
            dim, device=device, dtype=dtype, requires_grad=False
        )

    def forward(self, x: Tensor) -> Tensor:
        m = x.shape[0]
        if self.training:
            mean = ops.summation(x, 0) / m
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            mean = ops.broadcast_to(
                ops.reshape(mean, (1, self.dim)), (m, self.dim)
            )

            var = ops.summation((x - mean) ** 2, 0) / m
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var
            var = ops.broadcast_to(
                ops.reshape(var, (1, self.dim)), (m, self.dim)
            )
        else:
            mean = ops.broadcast_to(
                ops.reshape(self.running_mean, (1, self.dim)), (m, self.dim)
            )
            var = ops.broadcast_to(
                ops.reshape(self.running_var, (1, self.dim)), (m, self.dim)
            )
        x = (x - mean) / ((var + self.eps) ** 0.5)
        weight = ops.broadcast_to(
            ops.reshape(self.weight, (1, self.dim)), (m, self.dim)
        )
        bias = ops.broadcast_to(
            ops.reshape(self.bias, (1, self.dim)), (m, self.dim)
        )
        if self.training:
            return weight * x + bias
        return weight.data * x + bias.data
