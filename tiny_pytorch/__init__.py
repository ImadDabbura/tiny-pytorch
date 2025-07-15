"""Tiny PyTorch: A minimal deep learning framework.

Tiny PyTorch is a lightweight implementation of PyTorch-like functionality,
providing core tensor operations, automatic differentiation, neural network
modules, and optimization algorithms. It serves as both an educational tool
and a foundation for understanding deep learning frameworks.

The framework includes:
- Multi-dimensional tensors with automatic differentiation
- Neural network modules (Linear, Conv2d, BatchNorm, etc.)
- Optimization algorithms (SGD, Adam)
- Data loading utilities
- Multiple backend support (NumPy, CPU, CUDA)

Main Components
--------------
tensor : Core tensor data structure with automatic differentiation
ops : Tensor operations and their gradients
nn : Neural network modules and layers
optim : Optimization algorithms
data : Data loading and processing utilities
init : Tensor initialization functions
utils : Utility functions

Examples
--------
>>> import tiny_pytorch as tp
>>> x = tp.Tensor([1, 2, 3])
>>> y = x * 2
>>> y.backward()
>>> print(x.grad)
"""

from . import ops
from .backend_selection import *
from .tensor import Tensor
