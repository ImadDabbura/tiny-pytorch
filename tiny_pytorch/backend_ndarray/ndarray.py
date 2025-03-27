import numpy as np

from . import ndarray_backend_numpy


class BackendDevice:
    """Backend devive that wraps the implementation module for each device."""

    def __init__(self, name: str, module=None):
        self.name = name
        # Key attribute that will handle all ops on its device
        self.module = module

    def __eq__(self, other):
        # Two devives are equal if they have the same name
        return self.name == other.name

    def __repr__(self):
        return f"{self.name}()"

    def enabled(self):
        return self.module is not None

    def __getattr__(self, name):
        # All attempts to get attribute from device will be forwarded to the
        # module that implements the device's operations
        # i.e. device.op will become self.module.op
        return getattr(self.module, name)


def cpu_numpy():
    return BackendDevice("cpu_numpy", ndarray_backend_numpy)


def default_device():
    """Return cpu numpy backend."""
    return cpu_numpy()
