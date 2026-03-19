# Tensor

The core data structure of tiny-pytorch. A `Tensor` wraps an `NDArray` and builds a dynamic computation graph, enabling reverse-mode automatic differentiation (autograd).

```python
import tiny_pytorch as tp

x = tp.Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x * x).sum()
y.backward()
print(x.grad)  # [2.0, 4.0, 6.0]
```

::: tiny_pytorch.tensor
