from typing import Optional

import tiny_pytorch.nn as nn

from .tensor import Tensor


def ResidualBlock(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    device=None,
) -> nn.Residual:
    """
    Create a residual block with two ConvBN layers and a skip connection.

    This function creates a residual block that consists of two ConvBN layers
    followed by a residual connection that adds the input to the output.
    The residual connection helps with gradient flow and enables training
    of deeper networks.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolutional kernel.
    stride : int
        Stride of the convolution.
    device : Device, optional
        Device on which to place the parameters. Default is None (uses default device).

    Returns
    -------
    nn.Residual
        A residual block module that applies two ConvBN layers with a skip connection.

    Notes
    -----
    The residual block applies two ConvBN operations in sequence, then adds
    the original input to the result. This helps with gradient flow in deep networks.
    """
    main_path = nn.Sequential(
        nn.ConvBN(in_channels, out_channels, kernel_size, stride, device),
        nn.ConvBN(in_channels, out_channels, kernel_size, stride, device),
    )
    return nn.Residual(main_path)
