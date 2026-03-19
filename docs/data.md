# Data

Dataset and DataLoader utilities for building data pipelines. `Dataset` defines how to access individual samples, and `DataLoader` handles batching, shuffling, and applying transforms.

```python
from tiny_pytorch.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


loader = DataLoader(MyDataset(X_train, y_train), batch_size=32, shuffle=True)
for X_batch, y_batch in loader:
    ...
```

::: tiny_pytorch.data
