# Optimizers

Parameter update algorithms for training neural networks. Each optimizer takes a list of model parameters and updates them in-place based on their gradients.

```python
import tiny_pytorch as tp
import tiny_pytorch.nn as nn

model = nn.Linear(10, 1)
optimizer = tp.optim.Adam(model.parameters(), lr=0.001)

optimizer.zero_grad()
loss = model(tp.randn(4, 10)).sum()
loss.backward()
optimizer.step()
```

::: tiny_pytorch.optim
