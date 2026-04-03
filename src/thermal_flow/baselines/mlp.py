"""Standard MLP regression baseline for thermal inversion."""

import torch
import torch.nn as nn


class MLPInversion(nn.Module):
    """Multi-layer perceptron for direct signal-to-κ regression.

    Point estimate only (no uncertainty quantification).

    Args:
        input_dim: Dimension of input signal (n_freq * 2 for real+imag).
        output_dim: Dimension of output κ profile (n_layers).
        hidden_dims: List of hidden layer sizes.
        activation: Activation function name.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 120,
        output_dim: int = 100,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256, 256]

        act_fn = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}[activation]

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), act_fn(), nn.Dropout(dropout)])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Predict κ profile from 3-omega signal.

        Args:
            signal: Input signal, shape (batch, input_dim).

        Returns:
            Predicted κ profile, shape (batch, output_dim).
        """
        return self.net(signal)
