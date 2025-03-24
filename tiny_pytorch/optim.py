"""Optimization algorithms for neural networks.

This module implements various optimization algorithms commonly used in deep
learning, including Stochastic Gradient Descent (SGD) and Adam. These optimizers
are used to update model parameters during training to minimize the loss function.

The module provides a base Optimizer class that defines the common interface
for all optimizers, as well as concrete implementations of specific optimization
algorithms.

Classes
-------
Optimizer
    Base class that provides common optimizer functionality.
SGD
    Stochastic Gradient Descent optimizer with optional momentum.
Adam
    Adaptive Moment Estimation optimizer.

See Also
--------
[`tensor.Tensor`][tiny_pytorch.tensor.Tensor] : The Tensor class used for parameter optimization.

[`nn`][tiny_pytorch.nn] : Neural network modules whose parameters are optimized.
"""

import numpy as np

from . import init
from .tensor import Tensor


class Optimizer:
    """Base class for all optimizers.

    This class defines the basic interface and functionality that all optimizer
    implementations should follow. It provides common methods like step() for
    parameter updates and reset_grad() for gradient reset.

    Notes
    -----
    All optimizers should inherit from this base class and implement the step()
    method according to their specific optimization algorithm.

    See Also
    --------
    SGD : Stochastic Gradient Descent optimizer

    Adam : Adaptive Moment Estimation optimizer
    """

    def __init__(self, params):
        """
        Parameters
        ----------
        params : list
            List of parameters to optimize. Each parameter should be an instance
            of Tensor with requires_grad=True.
        """
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.

    Implements stochastic gradient descent (optionally with momentum).

    Notes
    -----
    The update rule for parameter `p` with gradient `g` is:

    With momentum:

        u = momentum * u + (1 - momentum) * g

        p = p * (1 - lr * weight_decay) - lr * u

    Without momentum:
        p = p * (1 - lr * weight_decay) - lr * g

    See Also
    --------
    Adam : Adaptive Moment Estimation optimizer
    """

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        """
        Parameters
        ----------
        params : list
            List of parameters to optimize. Each parameter should be an instance
            of Tensor with requires_grad=True.
        lr : float, optional
            Learning rate. Default: 0.01
        momentum : float, optional
            Momentum factor. Default: 0.0
        weight_decay : float, optional
            Weight decay (L2 penalty). Default: 0.0

        Notes
        -----
        When momentum is 0, this is equivalent to standard stochastic gradient
        descent. When momentum > 0, this implements momentum-based gradient
        descent which helps accelerate gradients vectors in the right directions.
        """
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
    """Adaptive Moment Estimation optimizer.

    Adam is an optimization algorithm that combines the benefits of two other
    extensions of stochastic gradient descent:
    - Adaptive Gradient Algorithm (AdaGrad)
    - Root Mean Square Propagation (RMSProp)

    It computes individual adaptive learning rates for different parameters
    from estimates of first and second moments of the gradients.

    The update rule for parameter p is:

        m = β₁ * m + (1 - β₁) * g         # First moment estimate

        v = β₂ * v + (1 - β₂) * g²        # Second moment estimate

        m̂ = m / (1 - β₁ᵗ)                 # Bias correction

        v̂ = v / (1 - β₂ᵗ)                 # Bias correction

        p = p - lr * m̂ / (√v̂ + ε)        # Update


    Notes
    -----
    The optimizer implements the Adam algorithm as described in
    "Adam: A Method for Stochastic Optimization" by Kingma and Ba (2014).

    See Also
    --------
    SGD : Stochastic Gradient Descent optimizer
    """

    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        """
        Parameters
        ----------
        params : list
            List of parameters to optimize.
        lr : float, optional
            Learning rate, by default 0.01
        beta1 : float, optional
            Exponential decay rate for first moment estimates, by default 0.9
        beta2 : float, optional
            Exponential decay rate for second moment estimates, by default 0.999
        eps : float, optional
            Small constant for numerical stability, by default 1e-8
        weight_decay : float, optional
            Weight decay (L2 penalty), by default 0.0

        """
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
