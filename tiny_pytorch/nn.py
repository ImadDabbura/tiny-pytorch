"""
This module provides a set of classes and functions for building neural networks.

Classes
-------
Module
    Base class for all neural network modules.
Parameter
    A special kind of tensor that represents parameters. It acts as a marker
    so modules can be able to identify learnable parameters. All `Parameter`
    tensors have require_grad set to True.
BatchNorm1d
    Batch normalization module.
LayerNorm1d
    Layer normalization module.
Dropout
    Dropout module.
Linear
    Linear transformation module.
Sequential
    Sequential container module.
Residual
    Residual connection module.
ReLU
    ReLU activation module.
SoftmaxLoss
    Softmax loss module.
Flatten
    Flatten module.
"""

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
    """
    Base class for all neural network modules. Your module should also subclass this.

    Attributes
    ----------
    training : bool
        Whether the module is in training mode or not.
    """

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
        """
        Returns
        -------
        list[Tensor]
            A list of tensors representing the parameters of the module.
        """
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def children(self) -> list["Module"]:
        """
        Return the list of child modules in the module.

        Returns
        -------
        list["Module"]
            List of child modules in the module.
        """
        return _child_modules(self.__dict__)

    def eval(self):
        """
        Sets the module in evaluation mode.

        This method sets the `training` attribute to `False`, which affects the behavior of certain modules like dropout and batch normalization. It also recursively sets the `training` attribute of all child modules.

        Notes
        -----
        This method is a no-op if the module is already in evaluation mode.
        """
        self.training = False

    def train(self):
        """
        Sets the module in training mode.

        This method sets the `training` attribute to `True`, which affects the behavior of certain modules like dropout and batch normalization. It also recursively sets the `training` attribute of all child modules.

        Notes
        -----
        This method is a no-op if the module is already in training mode.
        """
        self.training = True

    def __call__(self, *args, **kwargs):
        """
        Forward pass of the module.

        Returns
        -------
        Tensor
            The output tensor of the forward pass.
        """
        return self.forward(*args, **kwargs)


# TODO: Both weight and biases are initialized with Kaiming uniform
class Linear(Module):
    """
    Applies a linear transformation to the input data.

    Attributes
    ----------
    weight : Tensor
        The learnable weights of the module of shape `(in_features, out_features)`.
    bias : Tensor, optional
        The learnable bias of the module of shape `(1, out_features)`.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype="float32",
    ):
        """
        Parameters
        ----------
        in_features : int
            Size of each input sample.
        out_features : int
            Size of each output sample.
        bias : bool, optional
            If set to `False`, the layer will not learn an additive bias. Default is `True`.
        device : Device, optional
            Device on which to place the tensor. Default is CPU.
        dtype : str, optional
            Data type of the tensor. Default is "float32".
        """
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
    """
    Applies the rectified linear unit (ReLU) activation function element-wise.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor with ReLU activation applied element-wise.
    """

    def forward(self, x: Tensor) -> Tensor:
        return ops.ReLU()(x)


class Tanh(Module):
    """
    Applies the hyperbolic tangent (tanh) activation function element-wise.

    The tanh function maps any real-valued number to the range (-1, 1).
    It is defined as: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    Attributes
    ----------
    None
        This module has no learnable parameters.

    Examples
    --------
    >>> tanh = Tanh()
    >>> x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> output = tanh(x)
    >>> print(output)
    Tensor([-0.9640, -0.7616, 0.0000, 0.7616, 0.9640], device=cpu_numpy())
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the tanh activation function.

        Parameters
        ----------
        x : Tensor
            Input tensor of any shape.

        Returns
        -------
        Tensor
            Output tensor with the same shape as input, with tanh activation
            applied element-wise. Values are in the range (-1, 1).
        """
        return ops.tanh(x)


class Sequential(Module):
    """
    Applies a sequence of modules to the input.

    Parameters
    ----------
    *modules : Module
        A sequence of modules to apply to the input.

    Returns
    -------
    Tensor
        The output tensor after applying all modules in sequence.
    """

    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    """
    Computes the softmax loss between logits and labels.

    Parameters
    ----------
    logits : Tensor
        Input logits tensor.
    y : Tensor
        Ground truth labels tensor.

    Returns
    -------
    Tensor
        The softmax loss between logits and labels.
    """

    def forward(self, logits: Tensor, y: Tensor):
        m, k = logits.shape
        y_one_hot = init.one_hot(k, y.numpy().tolist())
        log_sum_exp = ops.BroadcastTo((m, k))(
            ops.Reshape((m, 1))(ops.LogSumExp((1,))(logits))
        )
        log_softmax = logits - log_sum_exp
        return -ops.Summation()(log_softmax * y_one_hot) / m


class LayerNorm1d(Module):
    """
    Applies layer normalization to the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor to apply layer normalization.
    dim : int
        Dimension to normalize.
    eps : float, optional
        Epsilon for numerical stability. Default is 1e-5.
    device : Device, optional
        Device on which to place the tensor. Default is CPU.
    dtype : str, optional
        Data type of the tensor. Default is "float32".

    Returns
    -------
    Tensor
        Normalized tensor.
    """

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
    """
    Flattens the input tensor into a 2D tensor.

    Parameters
    ----------
    X : Tensor
        Input tensor to be flattened.

    Returns
    -------
    Tensor
        Flattened tensor.
    """

    def forward(self, X):
        shape = X.shape
        return ops.Reshape((shape[0], reduce(mul, shape[1:])))(X)


class Dropout(Module):
    """
    Applies dropout to the input tensor.

    Parameters
    ----------
    p : float, optional
        Probability of an element to be dropped. Default is 0.5.

    Attributes
    ----------
    p : float
        Probability of an element to be dropped.

    Methods
    -------
    forward(x)
        Applies dropout to the input tensor `x`.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = init.randb(*x.shape, p=(1 - self.p))
            return (mask * x) / (1 - self.p)
        return x


class Residual(Module):
    """
    Applies a residual connection to the input tensor.

    Parameters
    ----------
    fn : Module
        The module to apply before adding the residual connection.

    Attributes
    ----------
    fn : Module
        The module to apply before adding the residual connection.

    Methods
    -------
    forward(x)
        Applies the residual connection to the input tensor `x`.
    """

    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x


class BatchNorm1d(Module):
    """
    Applies batch normalization to the input tensor.

    Parameters
    ----------
    dim : int
        Number of dimensions in the input tensor.
    eps : float, optional
        Value added to the denominator for numerical stability. Default is 1e-5.
    momentum : float, optional
        Momentum for the moving average. Default is 0.1.
    device : Device, optional
        Device on which to place the tensor. Default is CPU.
    dtype : str, optional
        Data type of the tensor. Default is "float32".

    Attributes
    ----------
    dim : int
        Number of dimensions in the input tensor.
    eps : float
        Value added to the denominator for numerical stability.
    momentum : float
        Momentum for the moving average.
    weight : Parameter
        Learnable weight parameter.
    bias : Parameter
        Learnable bias parameter.
    running_mean : Tensor
        Running mean of the input tensor.
    running_var : Tensor
        Running variance of the input tensor.

    Methods
    -------
    forward(x)
        Applies batch normalization to the input tensor `x`.
    """

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
