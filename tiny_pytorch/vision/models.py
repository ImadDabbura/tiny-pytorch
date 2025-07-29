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


class ResNet9(nn.Module):
    """
    ResNet-9: A lightweight ResNet architecture for image classification.

    ResNet-9 is a simplified version of the ResNet architecture designed for
    efficient training and inference on smaller datasets. It consists of
    convolutional layers with residual connections, followed by fully connected
    layers for classification.

    The architecture follows this pattern:
    1. Initial convolution (7x7, stride 4) to reduce spatial dimensions
    2. Multiple convolutional blocks with residual connections
    3. Global average pooling (via Flatten)
    4. Fully connected layers for classification

    Parameters
    ----------
    device : Device, optional
        Device on which to place the model parameters. Default is None (uses default device).

    Attributes
    ----------
    model : nn.Sequential
        The complete ResNet-9 model as a sequential container.

    Notes
    -----
    - Input is expected to be in NCHW format (batch, channels, height, width).
    - The model is designed for 10-class classification (e.g., CIFAR-10).
    - Uses residual connections to help with gradient flow in deeper layers.
    - The architecture progressively reduces spatial dimensions while increasing
      the number of channels.

    Examples
    --------
    >>> model = ResNet9()
    >>> x = Tensor.randn(32, 3, 32, 32)  # batch_size=32, channels=3, height=32, width=32
    >>> output = model(x)  # shape: (32, 10) - 10 class probabilities
    """

    def __init__(self, device=None) -> None:
        """
        Initialize the ResNet-9 model.

        Parameters
        ----------
        device : Device, optional
            Device on which to place the model parameters. Default is None (uses default device).
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvBN(3, 16, 7, 4, device=device),
            nn.ConvBN(16, 32, 3, 2, device=device),
            ResidualBlock(32, 32, 3, 1, device=device),
            nn.ConvBN(32, 64, 3, 2, device=device),
            nn.ConvBN(64, 128, 3, 2, device=device),
            ResidualBlock(128, 128, 3, 1, device=device),
            nn.Flatten(),
            nn.Linear(128, 128, device=device),
            nn.ReLU(),
            nn.Linear(128, 10, device=device),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ResNet-9 model.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, 3, height, width) in NCHW format.
            Typically used with 32x32 or 64x64 images.

        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, 10) containing class logits
            for 10-class classification.
        """
        return self.model(x)
