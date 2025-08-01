"""Neural network module for tiny-pytorch implementation.

This module provides a comprehensive set of classes and functions for building
neural networks. It includes fundamental building blocks like layers, activation
functions, normalization modules, and specialized components for various types
of neural network architectures.

The module is designed to work seamlessly with the automatic differentiation
system, allowing for easy construction and training of complex neural networks.
All modules inherit from the base `Module` class, providing consistent interfaces
for parameter management, training mode control, and forward pass computation.

Key Features
-----------
- Automatic parameter management and gradient tracking
- Training/evaluation mode switching
- Modular design for easy network construction
- Support for various neural network architectures
- Built-in activation functions and loss functions
- Normalization layers (BatchNorm, LayerNorm)
- Recurrent neural network components (RNN, LSTM)
- Convolutional neural network layers and composite blocks
- Embedding layers for sequence processing

Classes
-------
Module
    Base class for all neural network modules. Provides common functionality
    for parameter management, training mode control, and forward pass computation.
Parameter
    A special kind of tensor that represents learnable parameters. Acts as a marker
    so modules can identify trainable parameters. All Parameter tensors have
    require_grad set to True.
ReLU
    Rectified Linear Unit activation function.
Tanh
    Hyperbolic tangent activation function.
Sigmoid
    Sigmoid activation function.
Linear
    Linear transformation layer (fully connected layer).
Flatten
    Flattens the input tensor into a 2D tensor.
BatchNorm1d
    1D batch normalization layer for fully connected networks.
BatchNorm2d
    2D batch normalization layer for convolutional networks.
LayerNorm1d
    1D layer normalization layer.
Dropout
    Dropout layer for regularization during training.
Sequential
    Sequential container that applies modules in order.
Residual
    Residual connection that adds input to the output of a module.
SoftmaxLoss
    Softmax cross-entropy loss function.
Conv
    2D convolutional layer with support for padding and stride.
ConvBN
    Composite module combining convolution, batch normalization, and ReLU activation.
    Common building block in modern CNN architectures like ResNet.
RNNCell
    Single RNN cell with tanh or ReLU nonlinearity.
RNN
    Multi-layer RNN with tanh or ReLU nonlinearity.
LSTMCell
    Single LSTM cell with forget, input, and output gates.
LSTM
    Multi-layer LSTM network.
Embedding
    Embedding layer for converting indices to dense vectors.

Notes
-----
All modules support automatic differentiation through the tensor system.
Parameters are automatically tracked and gradients are computed during
backward passes. The training mode affects the behavior of certain modules
like Dropout and BatchNorm, which behave differently during training and
evaluation.

The module system is designed to be composable, allowing complex networks
to be built from simple building blocks. The Sequential and Residual
containers provide convenient ways to combine multiple modules.

Examples
--------
>>> import tiny_pytorch as tp
>>>
>>> # Create a simple feedforward network
>>> model = tp.nn.Sequential(
...     tp.nn.Linear(784, 128),
...     tp.nn.ReLU(),
...     tp.nn.Dropout(0.5),
...     tp.nn.Linear(128, 10)
... )
>>>
>>> # Create a convolutional network
>>> conv_model = tp.nn.Sequential(
...     tp.nn.Conv(3, 64, kernel_size=3),
...     tp.nn.BatchNorm2d(64),
...     tp.nn.ReLU(),
...     tp.nn.Flatten(),
...     tp.nn.Linear(64 * 28 * 28, 10)
... )
>>>
>>> # Create an RNN for sequence processing
>>> rnn = tp.nn.RNN(input_size=100, hidden_size=64, num_layers=2)
>>>
>>> # Use the model
>>> x = tp.Tensor.randn(32, 784)  # batch_size=32, features=784
>>> output = model(x)  # Forward pass
"""

from functools import reduce
from operator import mul
from typing import Any, Optional, Sequence

import numpy as np

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


class Sigmoid(Module):
    """
    Applies the sigmoid activation function element-wise.

    The sigmoid function maps any real-valued number to the range (0, 1).
    It is defined as: sigmoid(x) = 1 / (1 + e^(-x))

    The sigmoid function is commonly used in binary classification problems
    and as a gating mechanism in neural networks.

    Attributes
    ----------
    None
        This module has no learnable parameters.

    Examples
    --------
    >>> sigmoid = Sigmoid()
    >>> x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> output = sigmoid(x)
    >>> print(output)
    Tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808], device=cpu_numpy())
    """

    def __init__(self):
        """
        Initialize the Sigmoid module.

        This module has no learnable parameters and requires no initialization.
        """
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the sigmoid activation function.

        Parameters
        ----------
        x : Tensor
            Input tensor of any shape.

        Returns
        -------
        Tensor
            Output tensor with the same shape as input, with sigmoid activation
            applied element-wise. Values are in the range (0, 1).
        """
        return (1 + ops.exp(-x)) ** (-1)


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


class BatchNorm2d(BatchNorm1d):
    """
    Applies batch normalization to 2D input tensors.

    This module applies batch normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    "Batch Normalization: Accelerating Deep Network Training by Reducing Internal
    Covariate Shift".

    The input is expected to be in NCHW format (batch, channels, height, width).
    For each channel, this layer computes the mean and variance over the batch
    and spatial dimensions, then normalizes the input and applies learnable
    scale and shift parameters.

    Parameters
    ----------
    num_features : int
        Number of features/channels in the input tensor.
    eps : float, optional
        Value added to the denominator for numerical stability. Default is 1e-5.
    momentum : float, optional
        Momentum for the moving average. Default is 0.1.
    device : Device, optional
        Device on which to place the parameters. Default is None (uses default device).
    dtype : str, optional
        Data type of the parameters. Default is "float32".

    Attributes
    ----------
    num_features : int
        Number of features/channels in the input tensor.
    eps : float
        Value added to the denominator for numerical stability.
    momentum : float
        Momentum for the moving average.
    weight : Parameter
        Learnable weight parameter of shape (num_features,).
    bias : Parameter
        Learnable bias parameter of shape (num_features,).
    running_mean : Tensor
        Running mean of the input tensor of shape (num_features,).
    running_var : Tensor
        Running variance of the input tensor of shape (num_features,).

    Notes
    -----
    - Input is expected to be in NCHW format (batch, channels, height, width).
    - During training, this layer keeps a running estimate of its computed mean
      and variance, which is then used for normalization during evaluation.
    - The running estimates are kept with a default momentum of 0.1.
    - Internally converts to channel-last format for efficient computation,
      similar to PyTorch's implementation.

    Examples
    --------
    >>> bn = BatchNorm2d(64)
    >>> x = Tensor.randn(32, 64, 28, 28)  # batch_size=32, channels=64, height=28, width=28
    >>> output = bn(x)  # shape: (32, 64, 28, 28)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device=None,
        dtype: str = "float32",
    ) -> None:
        """
        Initialize the BatchNorm2d module.

        Parameters
        ----------
        num_features : int
            Number of features/channels in the input tensor.
        eps : float, optional
            Value added to the denominator for numerical stability. Default is 1e-5.
        momentum : float, optional
            Momentum for the moving average. Default is 0.1.
        device : Device, optional
            Device on which to place the parameters. Default is None (uses default device).
        dtype : str, optional
            Data type of the parameters. Default is "float32".
        """
        super().__init__(num_features, eps, momentum, device, dtype)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the 2D batch normalization.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, num_features, height, width) in NCHW format.

        Returns
        -------
        Tensor
            Normalized tensor of shape (batch_size, num_features, height, width) in NCHW format.
        """
        # It is more efficient to use channel-last in the case
        # of batchnorm (similar to pytorch)
        b, c_in, h, w = x.shape
        new_x = x.transpose((1, 2)).transpose((2, 3)).reshape((b, c_in, h, w))
        y = super().forward(new_x).reshape((b, c_in, h, w))
        return y.transpose((2, 3)).transpose((1, 2))


class ConvBN(Module):
    """
    A composite module that combines convolution, batch normalization, and ReLU activation.

    This module is a common building block in convolutional neural networks, particularly
    in architectures like ResNet. It applies a 2D convolution followed by batch normalization
    and ReLU activation in sequence. This combination helps with training stability and
    convergence speed.

    The module consists of three components applied in order:
    1. Conv: 2D convolutional layer
    2. BatchNorm2d: 2D batch normalization layer
    3. ReLU: Rectified Linear Unit activation function

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the convolution.
    kernel_size : int or tuple[int, int]
        Size of the convolving kernel. If a single int is provided, it is used
        for both height and width dimensions. Only square kernels are supported.
    stride : int or tuple[int, int], optional
        Stride of the convolution. If a single int is provided, it is used for
        both height and width dimensions. Default is 1.
    device : Device, optional
        Device on which to place the parameters. Default is None (uses default device).

    Attributes
    ----------
    conv : Conv
        The 2D convolutional layer.
    bn : BatchNorm2d
        The 2D batch normalization layer.
    relu : ReLU
        The ReLU activation function.

    Notes
    -----
    - Input is expected to be in NCHW format (batch, channels, height, width).
    - Output maintains the same format as input.
    - The convolution uses padding='same' to maintain spatial dimensions.
    - Batch normalization is applied per-channel across the batch and spatial dimensions.
    - ReLU activation is applied element-wise after batch normalization.

    Examples
    --------
    >>> convbn = ConvBN(3, 64, kernel_size=3, stride=1)
    >>> x = Tensor.randn(32, 3, 28, 28)  # batch_size=32, channels=3, height=28, width=28
    >>> output = convbn(x)  # shape: (32, 64, 28, 28)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        device=None,
    ) -> None:
        """
        Initialize the ConvBN module.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input image.
        out_channels : int
            Number of channels produced by the convolution.
        kernel_size : int or tuple[int, int]
            Size of the convolving kernel. If a single int is provided, it is used
            for both height and width dimensions.
        stride : int or tuple[int, int], optional
            Stride of the convolution. If a single int is provided, it is used for
            both height and width dimensions. Default is 1.
        device : Device, optional
            Device on which to place the parameters. Default is None (uses default device).
        """
        super().__init__()
        self.conv = Conv(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            device=device,
        )
        self.bn = BatchNorm2d(out_channels, device=device)
        self.relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ConvBN module.

        Applies convolution, batch normalization, and ReLU activation in sequence.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, in_channels, height, width) in NCHW format.

        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, out_channels, height, width) in NCHW format.
            The output has been processed through convolution, batch normalization, and ReLU.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Embedding(Module):
    """A lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.

    Parameters
    ----------
    vocab_sz : int
        Size of the dictionary of embeddings (number of unique tokens).
    embedding_dim : int
        The size of each embedding vector.
    device : Device, optional
        Device on which to place the embedding weights. Default is None (uses default device).
    dtype : str, optional
        Data type of the embedding weights. Default is "float32".

    Attributes
    ----------
    vocab_sz : int
        Size of the dictionary of embeddings.
    embedding_dim : int
        The size of each embedding vector.
    weight : Parameter
        The learnable embedding weights of shape `(vocab_sz, embedding_dim)`.
        Initialized from N(0, 1) distribution.

    Methods
    -------
    forward(x: Tensor) -> Tensor
        Maps word indices to embedding vectors.

    Examples
    --------
    >>> embedding = Embedding(1000, 128)
    >>> input_indices = Tensor([[1, 2, 3], [4, 5, 6]])  # shape: (seq_len, batch_size)
    >>> output = embedding(input_indices)  # shape: (seq_len, batch_size, 128)
    """

    def __init__(
        self,
        vocab_sz: int,
        embedding_dim: int,
        device=None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        self.vocab_sz = vocab_sz
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            init.randn(
                vocab_sz,
                embedding_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """Maps word indices to embedding vectors.

        This method converts input indices to one-hot vectors and then projects
        them to embedding vectors using the learned embedding weights.

        Parameters
        ----------
        x : Tensor
            Input tensor containing indices of shape `(seq_len, batch_size)`.
            Each element should be an integer index in the range [0, vocab_sz).

        Returns
        -------
        Tensor
            Output tensor of shape `(seq_len, batch_size, embedding_dim)` containing
            the corresponding embedding vectors for each input index.

        Notes
        -----
        The input indices are converted to one-hot vectors internally, then
        multiplied with the embedding weight matrix to produce the final embeddings.
        """
        T, B = x.shape
        x_one_hot = init.one_hot(
            self.vocab_sz, x, device=x.device, dtype=x.dtype
        )
        return (
            x_one_hot.reshape((T * B, self.vocab_sz)) @ self.weight
        ).reshape((T, B, self.embedding_dim))


class RNNCell(Module):
    """
    Applies a single RNN cell with a specified nonlinearity (tanh or ReLU).

    Parameters
    ----------
    input_size : int
        The number of expected features in the input X.
    hidden_size : int
        The number of features in the hidden state h.
    bias : bool, optional
        If False, then the layer does not use bias weights. Default is True.
    nonlinearity : str, optional
        The non-linearity to use. Can be either 'tanh' or 'relu'. Default is 'tanh'.
    device : Device, optional
        Device on which to place the weights. Default is None (uses default device).
    dtype : str, optional
        Data type of the weights. Default is "float32".

    Attributes
    ----------
    W_ih : Parameter
        The learnable input-hidden weights of shape (input_size, hidden_size).
    W_hh : Parameter
        The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
    bias_ih : Parameter or None
        The learnable input-hidden bias of shape (hidden_size,). None if bias is False.
    bias_hh : Parameter or None
        The learnable hidden-hidden bias of shape (hidden_size,). None if bias is False.
    nonlinearity : Module
        The nonlinearity module (Tanh or ReLU).
    device : Device or None
        Device on which the parameters are allocated.
    dtype : str
        Data type of the parameters.
    hidden_size : int
        The number of features in the hidden state h.

    Methods
    -------
    forward(X: Tensor, h: Tensor | None = None) -> Tensor
        Compute the next hidden state given input X and previous hidden state h.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
        device=None,
        dtype: str = "float32",
    ) -> None:
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.bias = bias
        self.hidden_size = hidden_size
        bound = np.sqrt(1 / hidden_size)
        self.W_ih = Parameter(
            init.rand(
                input_size,
                hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.bias_hh = Parameter(
                init.rand(
                    hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
        else:
            self.bias_ih = None
            self.bias_hh = None
        if nonlinearity == "tanh":
            self.nonlinearity = Tanh()
        elif nonlinearity == "relu":
            self.nonlinearity = ReLU()
        else:
            raise ValueError(
                "unsupported nonlinearity function. Only support ReLU and Tanh."
            )

    def forward(self, X: Tensor, h: Tensor | None = None) -> Tensor:
        """
        Compute the next hidden state for a batch of inputs.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (batch_size, input_size).
        h : Tensor or None, optional
            Initial hidden state for each element in the batch, of shape (batch_size, hidden_size). If None, defaults to zeros.

        Returns
        -------
        Tensor
            Next hidden state tensor of shape (batch_size, hidden_size).
        """
        batch_size, _ = X.shape
        if h is None:
            h = init.zeros(
                batch_size,
                self.hidden_size,
                device=self.device,
                dtype=self.dtype,
            )
        if self.bias:
            return self.nonlinearity(
                X @ self.W_ih
                + self.bias_ih.reshape((1, self.hidden_size)).broadcast_to(
                    (batch_size, self.hidden_size)
                )
                + h @ self.W_hh
                + self.bias_hh.reshape((1, self.hidden_size)).broadcast_to(
                    (batch_size, self.hidden_size)
                )
            )
        else:
            return self.nonlinearity(X @ self.W_ih + h @ self.W_hh)


class RNN(Module):
    """
    Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input x.
    hidden_size : int
        The number of features in the hidden state h.
    num_layers : int, optional
        Number of recurrent layers. Default is 1.
    bias : bool, optional
        If False, then the layer does not use bias weights. Default is True.
    nonlinearity : str, optional
        The non-linearity to use. Can be either 'tanh' or 'relu'. Default is 'tanh'.
    device : Device, optional
        Device on which to place the weights. Default is None (uses default device).
    dtype : str, optional
        Data type of the weights. Default is "float32".

    Attributes
    ----------
    rnn_cells : list of RNNCell
        List of RNNCell modules for each layer.
    hidden_size : int
        The number of features in the hidden state h.
    num_layers : int
        Number of recurrent layers.
    device : Device or None
        Device on which the parameters are allocated.
    dtype : str
        Data type of the parameters.

    Methods
    -------
    forward(X: Tensor, h0: Optional[Tensor] = None) -> tuple[Tensor, Tensor]
        Compute the output and final hidden state for a batch of input sequences.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        nonlinearity: str = "tanh",
        device=None,
        dtype: str = "float32",
    ) -> None:
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_cells = [
            (
                RNNCell(
                    input_size,
                    hidden_size,
                    bias=bias,
                    nonlinearity=nonlinearity,
                    device=device,
                    dtype=dtype,
                )
                if i == 0
                else RNNCell(
                    hidden_size,
                    hidden_size,
                    bias=bias,
                    nonlinearity=nonlinearity,
                    device=device,
                    dtype=dtype,
                )
            )
            for i in range(num_layers)
        ]

    def forward(
        self, X: "Tensor", h0: "Optional[Tensor]" = None
    ) -> tuple["Tensor", "Tensor"]:
        """
        Compute the output and final hidden state for a batch of input sequences.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (seq_len, batch_size, input_size) containing the features of the input sequence.
        h0 : Tensor or None, optional
            Initial hidden state for each element in the batch, of shape (num_layers, batch_size, hidden_size). If None, defaults to zeros.

        Returns
        -------
        output : Tensor
            Output tensor of shape (seq_len, batch_size, hidden_size) containing the output features (h_t) from the last layer of the RNN, for each t.
        h_n : Tensor
            Tensor of shape (num_layers, batch_size, hidden_size) containing the final hidden state for each element in the batch.
        """
        _, batch_size, _ = X.shape
        if h0 is None:
            h0_list = [
                init.zeros(
                    batch_size,
                    self.hidden_size,
                    device=self.device,
                    dtype=self.dtype,
                )
                for _ in range(self.num_layers)
            ]
        else:
            # h0: (num_layers, batch_size, hidden_size)
            h0_split = ops.split(h0, 0)
            h0_list = [h for h in h0_split]
        h_n = []
        X_split = ops.split(X, 0)
        inputs = [x for x in X_split]
        for num_layer in range(self.num_layers):
            h = h0_list[num_layer]
            for t, input in enumerate(inputs):
                h = self.rnn_cells[num_layer](input, h)
                inputs[t] = h
            h_n.append(h)
        # We'll detach history to avoid BPTT issues and keep the last hidden
        # state for each layer
        return ops.stack(inputs, 0), ops.stack(h_n, 0).detach()


class LSTMCell(Module):
    """
    A long short-term memory (LSTM) cell.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input X.
    hidden_size : int
        The number of features in the hidden state h.
    bias : bool, optional
        If False, then the layer does not use bias weights. Default is True.
    device : Device, optional
        Device on which to place the weights. Default is None (uses default device).
    dtype : str, optional
        Data type of the weights. Default is "float32".

    Attributes
    ----------
    W_ih : Parameter
        The learnable input-hidden weights, of shape (input_size, 4 * hidden_size).
    W_hh : Parameter
        The learnable hidden-hidden weights, of shape (hidden_size, 4 * hidden_size).
    bias_ih : Parameter or None
        The learnable input-hidden bias, of shape (4 * hidden_size,). None if bias is False.
    bias_hh : Parameter or None
        The learnable hidden-hidden bias, of shape (4 * hidden_size,). None if bias is False.
    hidden_size : int
        The number of features in the hidden state h.
    device : Device or None
        Device on which the parameters are allocated.
    dtype : str
        Data type of the parameters.

    Methods
    -------
    forward(X: Tensor, h: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, Tensor]
        Compute the next hidden and cell state given input X and previous states.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype: str = "float32",
    ) -> None:
        """
        A long short-term memory (LSTM) cell.

        Parameters
        ----------
        input_size : int
            The number of expected features in the input X.
        hidden_size : int
            The number of features in the hidden state h.
        bias : bool, optional
            If False, then the layer does not use bias weights. Default is True.
        device : Device, optional
            Device on which to place the weights. Default is None (uses default device).
        dtype : str, optional
            Data type of the weights. Default is "float32".

        Attributes
        ----------
        W_ih : Parameter
            The learnable input-hidden weights, of shape (input_size, 4 * hidden_size).
        W_hh : Parameter
            The learnable hidden-hidden weights, of shape (hidden_size, 4 * hidden_size).
        bias_ih : Parameter or None
            The learnable input-hidden bias, of shape (4 * hidden_size,). None if bias is False.
        bias_hh : Parameter or None
            The learnable hidden-hidden bias, of shape (4 * hidden_size,). None if bias is False.
        hidden_size : int
            The number of features in the hidden state h.
        device : Device or None
            Device on which the parameters are allocated.
        dtype : str
            Data type of the parameters.

        Methods
        -------
        forward(X: Tensor, h: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, Tensor]
            Compute the next hidden and cell state given input X and previous states.
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.bias = bias
        self.hidden_size = hidden_size
        bound = np.sqrt(1 / hidden_size)
        # Using one matrix for all gates/biases for efficiency
        self.W_ih = Parameter(
            init.rand(
                input_size,
                4 * hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                4 * hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    4 * hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.bias_hh = Parameter(
                init.rand(
                    4 * hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
        else:
            self.bias_ih = None
            self.bias_hh = None
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

    def forward(
        self, X: "Tensor", h: "tuple[Tensor, Tensor] | None" = None
    ) -> "tuple[Tensor, Tensor]":
        """
        Compute the next hidden and cell state for a batch of inputs.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (batch_size, input_size).
        h : tuple of (Tensor, Tensor) or None, optional
            Tuple of (h0, c0), where each is a tensor of shape (batch_size, hidden_size). If None, both default to zeros.

        Returns
        -------
        h_out : Tensor
            Next hidden state tensor of shape (batch_size, hidden_size).
        c_out : Tensor
            Next cell state tensor of shape (batch_size, hidden_size).
        """
        batch_size, _ = X.shape
        if h is None:
            h0, c0 = init.zeros(
                batch_size,
                self.hidden_size,
                device=self.device,
                dtype=self.dtype,
            ), init.zeros(
                batch_size,
                self.hidden_size,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            h0, c0 = h
        if self.bias:
            gates_all = (
                X @ self.W_ih
                + self.bias_ih.reshape((1, 4 * self.hidden_size)).broadcast_to(
                    (batch_size, 4 * self.hidden_size)
                )
                + h0 @ self.W_hh
                + self.bias_hh.reshape((1, 4 * self.hidden_size)).broadcast_to(
                    (batch_size, 4 * self.hidden_size)
                )
            )
        else:
            gates_all = X @ self.W_ih + h0 @ self.W_hh
        gates_all_split_tuple = ops.split(gates_all, axis=1)
        # To change outer container from TensorTuple to list
        gates_all_split = [g for g in gates_all_split_tuple]
        gates = []
        for i in range(4):
            gates.append(
                ops.stack(
                    gates_all_split[
                        i * self.hidden_size : (i + 1) * self.hidden_size
                    ],
                    axis=1,
                )
            )
        i, f, g, o = gates
        i, f, g, o = (
            self.sigmoid(i),
            self.sigmoid(f),
            self.tanh(g),
            self.sigmoid(o),
        )
        c_out = f * c0 + i * g
        h_out = o * self.tanh(c_out)
        return h_out, c_out


class LSTM(Module):
    """
    Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input x.
    hidden_size : int
        The number of features in the hidden state h.
    num_layers : int, optional
        Number of recurrent layers. Default is 1.
    bias : bool, optional
        If False, then the layer does not use bias weights. Default is True.
    device : Device, optional
        Device on which to place the weights. Default is None (uses default device).
    dtype : str, optional
        Data type of the weights. Default is "float32".

    Attributes
    ----------
    lstm_cells : list of LSTMCell
        List of LSTMCell modules for each layer.
    hidden_size : int
        The number of features in the hidden state h.
    num_layers : int
        Number of recurrent layers.
    device : Device or None
        Device on which the parameters are allocated.
    dtype : str
        Data type of the parameters.

    Methods
    -------
    forward(X: Tensor, h: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, tuple[Tensor, Tensor]]
        Compute the output and final hidden and cell states for a batch of input sequences.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        device=None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = [
            (
                LSTMCell(
                    input_size,
                    hidden_size,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                )
                if i == 0
                else LSTMCell(
                    hidden_size,
                    hidden_size,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                )
            )
            for i in range(num_layers)
        ]

    def forward(
        self, X: "Tensor", h: "tuple[Tensor, Tensor] | None" = None
    ) -> "tuple[Tensor, tuple[Tensor, Tensor]]":
        """
        Compute the output and final hidden and cell states for a batch of input sequences.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (seq_len, batch_size, input_size) containing the features of the input sequence.
        h : tuple of (Tensor, Tensor) or None, optional
            Tuple of (h0, c0), where each is a tensor of shape (num_layers, batch_size, hidden_size). If None, both default to zeros.

        Returns
        -------
        output : Tensor
            Output tensor of shape (seq_len, batch_size, hidden_size) containing the output features (h_t) from the last layer of the LSTM, for each t.
        (h_n, c_n) : tuple of Tensor
            Tuple of (h_n, c_n), each of shape (num_layers, batch_size, hidden_size) containing the final hidden and cell states for each element in the batch.
        """
        _, batch_size, _ = X.shape
        if h is None:
            h0, c0 = [
                init.zeros(
                    batch_size,
                    self.hidden_size,
                    device=self.device,
                    dtype=self.dtype,
                )
                for _ in range(self.num_layers)
            ], [
                init.zeros(
                    batch_size,
                    self.hidden_size,
                    device=self.device,
                    dtype=self.dtype,
                )
                for _ in range(self.num_layers)
            ]
        else:
            h0_split = ops.split(h[0], 0)
            c0_split = ops.split(h[1], 0)
            h0 = [x for x in h0_split]
            c0 = [x for x in c0_split]
        h_n, c_n = [], []
        X_split = ops.split(X, 0)
        inputs = [x for x in X_split]
        for num_layer in range(self.num_layers):
            h = h0[num_layer]
            c = c0[num_layer]
            for t, input in enumerate(inputs):
                h, c = self.lstm_cells[num_layer](input, (h, c))
                inputs[t] = h
            h_n.append(h)
            c_n.append(c)
        return ops.stack(inputs, 0), (
            ops.stack(h_n, 0).detach(),
            ops.stack(c_n, 0).detach(),
        )


class Conv(Module):
    """
    Multi-channel 2D convolutional layer.

    This module applies a 2D convolution over an input signal composed of several
    input planes. The input is expected to be in NCHW format (batch, channels, height, width)
    and the output will also be in NCHW format.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the convolution.
    kernel_size : int or tuple of int
        Size of the convolving kernel. If a single int is provided, it is used
        for both height and width dimensions. Only square kernels are supported.
    stride : int or tuple of int, optional
        Stride of the convolution. If a single int is provided, it is used for
        both height and width dimensions. Default is 1.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default is True.
    device : Device, optional
        Device on which to place the weights. Default is None (uses default device).
    dtype : str, optional
        Data type of the weights. Default is "float32".

    Attributes
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the convolution.
    kernel_size : int
        Size of the convolving kernel (square kernel).
    stride : int
        Stride of the convolution.
    padding : int
        Padding added to both sides of the input. Automatically calculated as
        (kernel_size - 1) // 2 to maintain same output size.
    weight : Parameter
        The learnable weights of the module of shape
        (kernel_size, kernel_size, in_channels, out_channels).
    bias : Parameter or None
        The learnable bias of the module of shape (out_channels,). None if bias is False.

    Notes
    -----
    - Only supports padding='same' (automatic padding to maintain output size).
    - No grouped convolution or dilation support.
    - Only supports square kernels.
    - Input and output are in NCHW format.

    Examples
    --------
    >>> conv = Conv(3, 64, kernel_size=3, stride=1)
    >>> x = Tensor.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    >>> output = conv(x)  # shape: (1, 64, 32, 32)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        bias: bool = True,
        device=None,
        dtype: str = "float32",
    ):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.weight = Parameter(
            init.kaiming_uniform(
                in_channels * kernel_size * kernel_size,
                out_channels * kernel_size * kernel_size,
                shape=(kernel_size, kernel_size, in_channels, out_channels),
                device=device,
            )
        )
        if bias:
            bias_bound = 1 / (in_channels * kernel_size**2) ** 0.5
            self.bias = Parameter(
                init.rand(
                    self.out_channels,
                    low=-bias_bound,
                    high=bias_bound,
                    device=device,
                )
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the 2D convolution.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, in_channels, height, width) in NCHW format.

        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, out_channels, height, width) in NCHW format.
        """
        nhwc_x = x.transpose((1, 2)).transpose((2, 3))
        out = ops.conv(
            nhwc_x, self.weight, stride=self.stride, padding=self.padding
        )
        if self.bias:
            out += self.bias.reshape(
                (1, 1, 1, self.out_channels)
            ).broadcast_to(out.shape)
        out = out.transpose((2, 3)).transpose((1, 2))
        return out
