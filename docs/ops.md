# Operators

Differentiable operations that form the building blocks of the computation graph. Each op implements a `compute` method (forward pass on NDArrays) and a `gradient` method (backward pass for autograd).

These are used internally by `Tensor` methods like `+`, `@`, `.reshape()`, etc.

```python
from tiny_pytorch import Tensor, ops

x = Tensor([[1.0, 2.0], [3.0, 4.0]])
y = ops.Summation(axes=(1,))(x)  # [3.0, 7.0]
```

::: tiny_pytorch.ops
