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
    """
    Generate a tensor filled with random numbers from uniform distribution
    between low and high.
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
    """
    Generate a tensor filled with random numbers from normal distribution
    with mean `mean` and variance `variance`.
    """
    device = cpu() if device is None else device
    array = device.randn(*shape) * std + mean
    return Tensor(
        array, device=device, dtype=dtype, requires_grad=requires_grad
    )


def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
    """Generate constant Tensor"""
    device = cpu() if device is None else device
    array = device.ones(shape, dtype=dtype) * c  # note: can change dtype
    return Tensor(
        array, device=device, dtype=dtype, requires_grad=requires_grad
    )


def ones(*shape, device=None, dtype="float32", requires_grad=False):
    """Generate all-ones Tensor"""
    return constant(
        *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def zeros(*shape, device=None, dtype="float32", requires_grad=False):
    """Generate all-zeros Tensor"""
    return constant(
        *shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def randb(*shape, p=0.5, device=None, dtype="bool", requires_grad=False):
    """Generate binary random Tensor"""
    device = cpu() if device is None else device
    array = device.rand(*shape) <= p
    return Tensor(
        array, device=device, dtype=dtype, requires_grad=requires_grad
    )


def one_hot(k, n, device=None, dtype="float32", requires_grad=False):
    """
    Generate one-hot encoding Tensor with `k` classes (columns) and `n`
    rows.
    """
    n = listify(n) if isinstance(n, Iterable) else n
    device = cpu() if device is None else device
    return Tensor(
        device.one_hot(k, n, dtype=dtype),
        device=device,
        requires_grad=requires_grad,
    )


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    a = gain * ((6 / (fan_in + fan_out)) ** 0.5)
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)
