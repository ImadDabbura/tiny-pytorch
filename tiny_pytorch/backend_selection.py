"""Backend selection logic for tiny-pytorch.

This module handles the selection and configuration of different array backends
for tiny-pytorch. It supports multiple backends including NumPy, CPU, and CUDA,
allowing users to choose the most appropriate backend for their use case.

The backend is selected via the TINY_PYTORCH_BACKEND environment variable:
- "nd": Uses the custom NDArray backend (default)
- "np": Uses NumPy as the backend

Environment Variables
--------------------
TINY_PYTORCH_BACKEND : str, optional
    The backend to use. Options are "nd" (default) or "np".

Raises
------
RuntimeError
    If an unknown backend is specified.

Examples
--------
>>> import os
>>> os.environ["TINY_PYTORCH_BACKEND"] = "np"
>>> import tiny_pytorch  # Will use NumPy backend
"""

import os

BACKEND = os.environ.get("TINY_PYTORCH_BACKEND", "nd")


if BACKEND == "nd":
    from . import backend_ndarray as array_api
    from .backend_ndarray import BackendDevice as Device
    from .backend_ndarray import (
        all_devices,
        cpu,
        cpu_numpy,
        cuda,
        default_device,
    )

    NDArray = array_api.NDArray
elif BACKEND == "np":
    import numpy as array_api

    from .backend_numpy import Device, all_devices, cpu, default_device

    NDArray = array_api.ndarray
else:
    raise RuntimeError("Unknown tiny-pytorch array backend {BACKEND}")
