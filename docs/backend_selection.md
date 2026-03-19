# Backend Selection

Controls which array backend tiny-pytorch uses. Set the `TINY_PYTORCH_BACKEND` environment variable before importing to switch backends.

- `"nd"` (default) -- Custom NDArray backend with CPU (C++) and CUDA support
- `"np"` -- NumPy backend (useful for debugging, supports float64)

```python
import os

os.environ["TINY_PYTORCH_BACKEND"] = "np"
import tiny_pytorch  # Will use NumPy backend
```

::: tiny_pytorch.backend_selection
