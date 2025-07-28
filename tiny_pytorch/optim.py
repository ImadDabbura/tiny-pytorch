"""Optimization algorithms for neural networks.

This module implements various optimization algorithms commonly used in deep
learning, including Stochastic Gradient Descent (SGD) and Adam. These optimizers
are used to update model parameters during training to minimize the loss function
through gradient-based optimization.

The module provides a base Optimizer class that defines the common interface
for all optimizers, as well as concrete implementations of specific optimization
algorithms. All optimizers work seamlessly with the Tensor system and automatic
differentiation capabilities.

Key Features
-----------
- Gradient-based optimization algorithms
- Support for momentum and weight decay
- Adaptive learning rate methods
- Automatic parameter updates
- Integration with Tensor gradients
- Memory-efficient optimization

Classes
-------
Optimizer
    Base class that provides common optimizer functionality.
    Defines the interface for parameter updates and gradient management.
SGD : Optimizer
    Stochastic Gradient Descent optimizer with optional momentum.
    Implements the classic gradient descent algorithm with support for
    momentum and weight decay regularization.
Adam : Optimizer
    Adaptive Moment Estimation optimizer.
    Implements the Adam algorithm with adaptive learning rates for each
    parameter, combining the benefits of AdaGrad and RMSprop.

Notes
-----
All optimizers in this module work with the Tensor system and automatically
handle gradient computation and parameter updates. The optimizers expect
parameters to be Tensor objects with gradients computed through the
automatic differentiation system.

The optimization algorithms implement various techniques to improve training
stability and convergence speed:
- Momentum helps accelerate convergence in relevant directions
- Weight decay provides regularization to prevent overfitting
- Adaptive learning rates (in Adam) help handle sparse gradients

Optimizers automatically handle gradient accumulation and parameter updates,
making them easy to use in training loops.

Examples
--------
>>> import tiny_pytorch as tp
>>>
>>> # Create a simple model
>>> model = tp.nn.Sequential(
...     tp.nn.Linear(784, 128),
...     tp.nn.ReLU(),
...     tp.nn.Linear(128, 10)
... )
>>>
>>> # Create an optimizer
>>> optimizer = tp.optim.SGD(
...     model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
... )
>>>
>>> # Training loop
>>> for epoch in range(num_epochs):
...     for batch_x, batch_y in dataloader:
...         # Forward pass
...         output = model(batch_x)
...         loss = tp.nn.SoftmaxLoss()(output, batch_y)
...
...         # Backward pass
...         loss.backward()
...
...         # Update parameters
...         optimizer.step()
...         optimizer.reset_grad()
>>>
>>> # Using Adam optimizer
>>> adam_optimizer = tp.optim.Adam(
...     model.parameters(), lr=0.001, beta1=0.9, beta2=0.999
... )

See Also
--------
tensor.Tensor : The Tensor class used for parameter optimization.
nn : Neural network modules whose parameters are optimized.
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
