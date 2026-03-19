# NDArray

The low-level strided N-dimensional array that underlies every `Tensor`. Provides a unified interface over multiple backends (NumPy, CPU via C++, CUDA) using a flat data buffer with shape/stride metadata.

```python
from tiny_pytorch.backend_ndarray import ndarray, cpu_numpy

device = cpu_numpy()
a = ndarray.NDArray.make((2, 3), device=device)
```

::: tiny_pytorch.backend_ndarray.ndarray
