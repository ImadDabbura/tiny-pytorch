# Optimizers

Parameter update algorithms for training neural networks. Each optimizer takes a list of model parameters and updates them in-place based on their gradients.

```python
import tiny_pytorch.nn as nn
import tiny_pytorch.optim as optim
from tiny_pytorch import init

model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

optimizer.reset_grad()
loss = model(init.randn(4, 10)).sum()
loss.backward()
optimizer.step()
```

::: tiny_pytorch.optim
