from typing import Optional, Tuple, Union

from .. import nn
from ..tensor import Tensor


class LanguageModel(nn.Module):
    """
    A language model for sequence prediction tasks.

    This module implements a complete language model architecture consisting of
    an embedding layer, a sequence model (RNN or LSTM), and a linear output layer.
    It is designed for tasks like next-word prediction, text generation, and
    other sequence modeling applications.

    The model architecture follows this pattern:
    1. Embedding layer: Converts input token indices to dense vectors
    2. Sequence model: Processes the embedded sequence (RNN or LSTM)
    3. Linear layer: Projects the final hidden states to vocabulary logits

    Parameters
    ----------
    num_embeddings : int
        The size of the vocabulary (number of unique tokens).
    embedding_dim : int
        The dimensionality of the embedding vectors.
    hidden_size : int
        The number of features in the hidden state of the sequence model.
    num_layers : int, optional
        Number of layers in the RNN or LSTM. Default is 1.
    seq_model : str, optional
        Type of sequence model to use. Must be either 'rnn' or 'lstm'. Default is 'rnn'.
    device : Device, optional
        Device on which to place the model parameters. Default is None (uses default device).
    dtype : str, optional
        Data type of the model parameters. Default is "float32".

    Attributes
    ----------
    output_size : int
        The size of the vocabulary.
    hidden_size : int
        The number of features in the hidden state.
    embed : nn.Embedding
        The embedding layer that converts token indices to vectors.
    seq_model : nn.RNN or nn.LSTM
        The sequence model (RNN or LSTM) for processing the embedded sequence.
    linear : nn.Linear
        The output linear layer that projects to vocabulary logits.

    Notes
    -----
    - Input sequences should be provided as token indices in shape (seq_len, batch_size).
    - The model outputs logits for next-token prediction at each position.
    - Supports both RNN and LSTM sequence models with configurable layers.
    - The embedding layer maps from vocabulary size to embedding dimension.
    - The linear layer projects from hidden size back to vocabulary size.

    Examples
    --------
    >>> model = LanguageModel(
    ...     embedding_size=128,
    ...     output_size=1000,  # vocabulary size
    ...     hidden_size=256,
    ...     num_layers=2,
    ...     seq_model='lstm'
    ... )
    >>>
    >>> # Input: (seq_len=10, batch_size=32)
    >>> x = Tensor.randint(0, 1000, (10, 32))
    >>>
    >>> # Forward pass
    >>> logits, hidden = model(x)
    >>> print(logits.shape)  # (320, 1000) - (seq_len*batch_size, vocab_size)
    >>> print(hidden[0].shape if isinstance(hidden, tuple) else hidden.shape)
    >>> # (2, 32, 256) - (num_layers, batch_size, hidden_size)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        seq_model: str = "rnn",
        device=None,
        dtype: str = "float32",
    ) -> None:
        """
        Initialize the LanguageModel.

        Parameters
        ----------
        num_embeddings : int
            The size of the vocabulary (number of unique tokens).
        embedding_dim : int
            The dimensionality of the embedding vectors.
        hidden_size : int
            The number of features in the hidden state of the sequence model.
        num_layers : int, optional
            Number of layers in the RNN or LSTM. Default is 1.
        seq_model : str, optional
            Type of sequence model to use. Must be either 'rnn' or 'lstm'. Default is 'rnn'.
        device : Device, optional
            Device on which to place the model parameters. Default is None (uses default device).
        dtype : str, optional
            Data type of the model parameters. Default is "float32".

        Raises
        ------
        AssertionError
            If seq_model is not 'rnn' or 'lstm'.
        """
        super().__init__()
        assert seq_model in [
            "rnn",
            "lstm",
        ], "Unsupported sequence model. Must be rnn or lstm."
        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(
            num_embeddings, embedding_dim, device=device, dtype=dtype
        )
        if seq_model == "rnn":
            self.seq_model = nn.RNN(
                embedding_dim,
                hidden_size,
                num_layers=num_layers,
                device=device,
                dtype=dtype,
            )
        else:
            self.seq_model = nn.LSTM(
                embedding_dim,
                hidden_size,
                num_layers=num_layers,
                device=device,
                dtype=dtype,
            )
        self.linear = nn.Linear(
            hidden_size, num_embeddings, device=device, dtype=dtype
        )

    def forward(
        self,
        x: Tensor,
        h: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
    ) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, Tensor]]]:
        """
        Forward pass of the language model.

        Given an input sequence of token indices, returns logits for next-token
        prediction along with the final hidden state from the sequence model.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (seq_len, batch_size) containing token indices.
            Each element should be an integer index in the range [0, output_size).
        h : Tensor or tuple of (Tensor, Tensor) or None, optional
            Initial hidden state for the sequence model.
            - For RNN: Tensor of shape (num_layers, batch_size, hidden_size)
            - For LSTM: Tuple of (h0, c0), each of shape (num_layers, batch_size, hidden_size)
            - If None, defaults to zeros for RNN or (zeros, zeros) for LSTM.

        Returns
        -------
        logits : Tensor
            Output tensor of shape (seq_len * batch_size, output_size) containing
            logits for next-token prediction at each position in the sequence.
        hidden : Tensor or tuple of (Tensor, Tensor)
            Final hidden state from the sequence model.
            - For RNN: Tensor of shape (num_layers, batch_size, hidden_size)
            - For LSTM: Tuple of (h_n, c_n), each of shape (num_layers, batch_size, hidden_size)

        Notes
        -----
        The output logits are flattened across the sequence dimension, so each
        position in the sequence contributes batch_size predictions. This is
        useful for training with cross-entropy loss where each position is
        treated as a separate prediction task.
        """
        seq_len, bs = x.shape
        x = self.embed(x)
        x, h = self.seq_model(x, h)
        x = self.linear(x.reshape((seq_len * bs, self.hidden_size)))
        return x, h
