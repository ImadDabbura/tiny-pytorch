"""
This module provides functions for initializing tensors with different types of values.
It includes functions for generating tensors filled with ones, zeros, random numbers, binary random numbers,
and one-hot encoded tensors. It also provides functions for initializing weights using Xavier/Glorot uniform
and Kaiming/He uniform and normal distributions.

Functions
---------
ones
    Generate a tensor filled with ones.
zeros
    Generate a tensor filled with zeros.
rand
    Generate a tensor filled with random numbers from a uniform distribution.
randb
    Generate a binary random tensor.
one_hot
    Generate a one-hot encoded tensor.
xavier_uniform
    Initialize weights using Xavier/Glorot uniform initialization.
kaiming_uniform
    Initialize weights using Kaiming/He uniform initialization.
kaiming_normal
    Initialize weights using Kaiming/He normal initialization.
"""

from collections.abc import Iterable

from .device import cpu
from .tensor import Tensor
from .utils import listify


def rand(
    *shape,
    low=0.0,
    high=1.0,
    device=None,
    dtype="float32",
    requires_grad=False,
):
    """Generate a tensor filled with random numbers from uniform distribution.

    Parameters
    ----------
    *shape : int
        Shape of the output tensor.
    low : float, optional
        Lower bound of the uniform distribution. Default is 0.0.
    high : float, optional
        Upper bound of the uniform distribution. Default is 1.0.
    device : Device, optional
        Device on which to place the tensor. Default is CPU.
    dtype : str, optional
        Data type of the tensor. Default is "float32".
    requires_grad : bool, optional
        If True, tensor will track gradients. Default is False.

    Returns
    -------
    Tensor
        Tensor of specified shape filled with random values from U(low, high).

    Examples
    --------
    >>> rand(2, 3)  # 2x3 tensor with values in [0,1]
    >>> rand(4, low=-1, high=1)  # 4-element tensor with values in [-1,1]
    >>> rand(2, 2, device=gpu(), dtype="float64")  # 2x2 tensor on GPU
    """
    device = cpu() if device is None else device
    array = device.rand(*shape) * (high - low) + low
    return Tensor(
        array, device=device, dtype=dtype, requires_grad=requires_grad
    )


def randn(
    *shape,
    mean=0.0,
    std=1.0,
    device=None,
    dtype="float32",
    requires_grad=False,
):
    """Generate a tensor filled with random numbers from normal distribution.

    Parameters
    ----------
    *shape : int
        Shape of the output tensor.
    mean : float, optional
        Mean of the normal distribution. Default is 0.0.
    std : float, optional
        Standard deviation of the normal distribution. Default is 1.0.
    device : Device, optional
        Device on which to place the tensor. Default is CPU.
    dtype : str, optional
        Data type of the tensor. Default is "float32".
    requires_grad : bool, optional
        If True, tensor will track gradients. Default is False.

    Returns
    -------
    Tensor
        Tensor of specified shape filled with random values from N(mean, std^2).

    Examples
    --------
    >>> randn(2, 3)  # 2x3 tensor from standard normal
    >>> randn(4, mean=5, std=0.1)  # 4-element tensor with mean 5, std 0.1
    >>> randn(2, 2, device=gpu(), dtype="float64")  # 2x2 tensor on GPU
    """
    device = cpu() if device is None else device
    array = device.randn(*shape) * std + mean
    return Tensor(
        array, device=device, dtype=dtype, requires_grad=requires_grad
    )


def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
    """Generate a tensor filled with a constant value.

    Parameters
    ----------
    *shape : int
        Shape of the output tensor.
    c : float, optional
        Constant value to fill the tensor with. Default is 1.0.
    device : Device, optional
        Device on which to place the tensor. Default is CPU.
    dtype : str, optional
        Data type of the tensor. Default is "float32".
    requires_grad : bool, optional
        If True, tensor will track gradients. Default is False.

    Returns
    -------
    Tensor
        Tensor of specified shape filled with constant value c.

    Examples
    --------
    >>> constant(2, 3, c=5)  # 2x3 tensor filled with 5
    >>> constant(4, c=3.14)  # 4-element tensor filled with pi
    >>> constant(2, 2, device=gpu(), c=2.0)  # 2x2 tensor on GPU filled with 2
    """
    device = cpu() if device is None else device
    array = device.ones(shape, dtype=dtype) * c  # note: can change dtype
    return Tensor(
        array, device=device, dtype=dtype, requires_grad=requires_grad
    )


def ones(*shape, device=None, dtype="float32", requires_grad=False):
    """Generate a tensor filled with ones.

    Parameters
    ----------
    *shape : int
        Shape of the output tensor.
    device : Device, optional
        Device on which to place the tensor. Default is CPU.
    dtype : str, optional
        Data type of the tensor. Default is "float32".
    requires_grad : bool, optional
        If True, tensor will track gradients. Default is False.

    Returns
    -------
    Tensor
        Tensor of specified shape filled with ones.

    Examples
    --------
    >>> ones(2, 3)  # 2x3 tensor filled with ones
    >>> ones(4, device=gpu())  # 4-element tensor of ones on GPU
    >>> ones(2, 2, dtype="float64")  # 2x2 tensor of ones with float64 dtype
    """
    return constant(
        *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def zeros(*shape, device=None, dtype="float32", requires_grad=False):
    """Generate a tensor filled with zeros.

    Parameters
    ----------
    *shape : int
        Shape of the output tensor.
    device : Device, optional
        Device on which to place the tensor. Default is CPU.
    dtype : str, optional
        Data type of the tensor. Default is "float32".
    requires_grad : bool, optional
        If True, tensor will track gradients. Default is False.

    Returns
    -------
    Tensor
        Tensor of specified shape filled with zeros.

    Examples
    --------
    >>> zeros(2, 3)  # 2x3 tensor filled with zeros
    >>> zeros(4, device=gpu())  # 4-element tensor of zeros on GPU
    >>> zeros(2, 2, dtype="float64")  # 2x2 tensor of zeros with float64 dtype
    """
    return constant(
        *shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def randb(*shape, p=0.5, device=None, dtype="bool", requires_grad=False):
    """Generate a binary random tensor.

    Parameters
    ----------
    *shape : int
        Shape of the output tensor.
    p : float, optional
        Probability of generating 1. Default is 0.5.
    device : Device, optional
        Device on which to place the tensor. Default is CPU.
    dtype : str, optional
        Data type of the tensor. Default is "bool".
    requires_grad : bool, optional
        If True, tensor will track gradients. Default is False.

    Returns
    -------
    Tensor
        Binary tensor of specified shape where each element is 1 with
        probability p and 0 with probability (1-p).

    Examples
    --------
    >>> randb(2, 3)  # 2x3 binary tensor with p=0.5
    >>> randb(4, p=0.8)  # 4-element tensor, 80% chance of 1s
    >>> randb(2, 2, device=gpu())  # 2x2 binary tensor on GPU
    """
    device = cpu() if device is None else device
    array = device.rand(*shape) <= p
    return Tensor(
        array, device=device, dtype=dtype, requires_grad=requires_grad
    )


def one_hot(k, n, device=None, dtype="float32", requires_grad=False):
    """Generate a one-hot encoded tensor.

    Parameters
    ----------
    k : int
        Number of classes (width of one-hot encoding).
    n : int or Iterable[int]
        Number of samples (rows) or shape of output tensor.
    device : Device, optional
        Device on which to place the tensor. Default is CPU.
    dtype : str, optional
        Data type of the tensor. Default is "float32".
    requires_grad : bool, optional
        If True, tensor will track gradients. Default is False.

    Returns
    -------
    Tensor
        One-hot encoded tensor with shape (n, k) if n is int,
        or (*n, k) if n is Iterable.

    Examples
    --------
    >>> one_hot(3, 2)  # 2x3 tensor with one-hot rows
    >>> one_hot(4, [2,3])  # 2x3x4 tensor with one-hot encodings
    >>> one_hot(2, 5, device=gpu())  # 5x2 one-hot tensor on GPU
    """
    n = listify(n) if isinstance(n, Iterable) else n
    device = cpu() if device is None else device
    return Tensor(
        device.one_hot(k, n, dtype=dtype),
        device=device,
        requires_grad=requires_grad,
    )


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    """Initialize weights using Xavier/Glorot uniform initialization.

    Parameters
    ----------
    fan_in : int
        Number of input features.
    fan_out : int
        Number of output features.
    gain : float, optional
        Scaling factor for the bounds of the uniform distribution.
        Default is 1.0.
    **kwargs
        Additional arguments passed to rand().

    Returns
    -------
    Tensor
        Tensor initialized with values from uniform distribution
        U(-a, a) where a = gain * sqrt(6/(fan_in + fan_out)).

    Notes
    -----
    This initialization helps maintain variance of activations and gradients
    across layers in deep networks. The gain parameter can be adjusted for
    different activation functions.

    References
    ----------
    Glorot, X. & Bengio, Y. (2010). Understanding the difficulty of training
    deep feedforward neural networks. In AISTATS.

    Examples
    --------
    >>> xavier_uniform(10, 5)  # 10x5 tensor with Xavier initialization
    >>> xavier_uniform(20, 10, gain=2.0)  # Scaled initialization
    >>> xavier_uniform(5, 5, device=gpu())  # Initialize on GPU
    """
    a = gain * ((6 / (fan_in + fan_out)) ** 0.5)
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    """Initialize weights using Xavier/Glorot normal initialization.

    Parameters
    ----------
    fan_in : int
        Number of input features.
    fan_out : int
        Number of output features.
    gain : float, optional
        Scaling factor for the standard deviation of the normal distribution.
        Default is 1.0.
    **kwargs
        Additional arguments passed to randn().

    Returns
    -------
    Tensor
        Tensor initialized with values from normal distribution
        N(mean, std^2) where mean = 0 and std = gain * sqrt(2/(fan_in + fan_out)).

    Notes
    -----
    This initialization helps maintain variance of activations and gradients
    across layers in deep networks. The gain parameter can be adjusted for
    different activation functions.

    References
    ----------
    Glorot, X. & Bengio, Y. (2010). Understanding the difficulty of training
    deep feedforward neural networks. In AISTATS.

    Examples
    --------
    >>> xavier_normal(10, 5)  # 10x5 tensor with Xavier initialization
    >>> xavier_normal(20, 10, gain=2.0)  # Scaled initialization
    >>> xavier_normal(5, 5, device=gpu())  # Initialize on GPU
    """
    std = gain * ((2 / (fan_in + fan_out)) ** 0.5)
    return randn(fan_in, fan_out, std=std, **kwargs)


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    """Initialize weights using Kaiming/He uniform initialization.

    Parameters
    ----------
    fan_in : int
        Number of input features.
    fan_out : int
        Number of output features.
    nonlinearity : str, optional
        The non-linear function (or activation function) used in the model.
        Default is "relu".
    **kwargs
        Additional arguments passed to rand().

    Returns
    -------
    Tensor
        Tensor initialized with values from uniform distribution
        U(-bound, bound) where bound = sqrt(6 / fan_in).

    Notes
    -----
    This initialization is designed to work with the rectified linear unit (ReLU)
    activation function. It sets the weights to be zero-mean and have a standard
    deviation of sqrt(2 / fan_in).

    References
    ----------
    He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers:
    Surpassing human-level performance on ImageNet classification. In ICCV.

    Examples
    --------
    >>> kaiming_uniform(10, 5)  # 10x5 tensor with Kaiming initialization
    >>> kaiming_uniform(20, 10, nonlinearity="tanh")  # Initialize for tanh
    >>> kaiming_uniform(5, 5, device=gpu())  # Initialize on GPU
    """
    assert nonlinearity == "relu", "Only relu supported currently"
    bound = (6 / fan_in) ** 0.5
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    """
    Initialize weights using Kaiming/He normal initialization.

    Parameters
    ----------
    fan_in : int
        Number of input features.
    fan_out : int
        Number of output features.
    nonlinearity : str, optional
        The non-linear function (or activation function) used in the model.
        Default is "relu".
    **kwargs
        Additional arguments passed to randn().

    Returns
    -------
    Tensor
        Tensor initialized with values from normal distribution
        N(0, std^2) where std = sqrt(2 / fan_in).

    Notes
    -----
    This initialization is designed to work with the rectified linear unit (ReLU)
    activation function. It sets the weights to be zero-mean and have a standard
    deviation of sqrt(2 / fan_in).

    References
    ----------
    He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers:
    Surpassing human-level performance on ImageNet classification. In ICCV.
    """
    assert nonlinearity == "relu", "Only relu supported currently"
    std = (2 / fan_in) ** 0.5
    return randn(fan_in, fan_out, std=std, **kwargs)
