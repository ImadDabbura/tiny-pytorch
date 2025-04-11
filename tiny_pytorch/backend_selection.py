"""Logic for backend selection"""

import os

BACKEND = os.environ.get("NEEDLE_BACKEND", "np")


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
    raise RuntimeError("Unknown needle array backend %s" % BACKEND)
