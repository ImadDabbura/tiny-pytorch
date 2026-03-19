# Initialization

Functions for creating and initializing tensors. Includes standard fills (`zeros`, `ones`, `rand`) and weight initialization schemes (`Xavier`, `Kaiming`) used when constructing neural network layers.

```python
import tiny_pytorch.init as init

w = init.kaiming_uniform(128, 64)  # (128, 64) tensor
b = init.zeros(64)  # (64,) tensor
x = init.randn(32, 128)  # (32, 128) tensor from N(0, 1)
```

::: tiny_pytorch.init
