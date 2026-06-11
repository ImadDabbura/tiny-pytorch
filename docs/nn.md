# Neural Networks

Neural network layers and loss functions built on top of the `Tensor` and `ops` modules. Follows the `Module` pattern familiar from PyTorch — subclass `Module`, implement `forward`, and compose layers freely.

```python
import tiny_pytorch.nn as nn
from tiny_pytorch import init

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
x = init.randn(32, 784)
out = model(x)  # (32, 10)
```

::: tiny_pytorch.nn
