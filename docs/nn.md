# Neural Networks

Neural network layers and loss functions built on top of the `Tensor` and `ops` modules. Follows the `Module` pattern familiar from PyTorch — subclass `Module`, implement `forward`, and compose layers freely.

```python
import tiny_pytorch as tp
import tiny_pytorch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
x = tp.randn(32, 784)
out = model(x)  # (32, 10)
```

::: tiny_pytorch.nn
