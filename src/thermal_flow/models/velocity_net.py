"""Velocity field network for Conditional Flow Matching.

Architecture: MLP with sinusoidal time embedding and FiLM conditioning
on the measurement signal. Relatively compact (~10^5-10^6 parameters)
since the input/output dimensions are ~60-100.
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for the time variable t ∈ [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed scalar time into a vector.

        Args:
            t: Time values, shape (batch,) or (batch, 1).

        Returns:
            Time embedding, shape (batch, dim).
        """
        t = t.view(-1)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer.

    Applies affine transformation conditioned on external input:
        output = γ(condition) * x + β(condition)
    """

    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()
        self.scale = nn.Linear(condition_dim, feature_dim)
        self.shift = nn.Linear(condition_dim, feature_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma = self.scale(condition)
        beta = self.shift(condition)
        return gamma * x + beta


class VelocityNet(nn.Module):
    """MLP velocity field v_θ(x_t, t, y) for Conditional Flow Matching.

    Predicts the velocity field that transports noise x_0 ~ N(0,I) to
    the target κ(z) distribution, conditioned on the measurement signal y.

    Args:
        x_dim: Dimension of the state x (= n_layers for κ profile).
        y_dim: Dimension of the condition y (= 2 * n_freq for signal).
        hidden_dims: List of hidden layer dimensions.
        time_embed_dim: Dimension of the sinusoidal time embedding.
        activation: Activation function name.
        condition_method: 'film' or 'concat'.
    """

    def __init__(
        self,
        x_dim: int = 100,
        y_dim: int = 120,
        hidden_dims: list[int] | None = None,
        time_embed_dim: int = 128,
        activation: str = "silu",
        condition_method: str = "film",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512, 512]

        self.condition_method = condition_method
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        act_fn = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}[activation]

        # Condition encoder
        cond_dim = time_embed_dim + y_dim
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, hidden_dims[0]),
            act_fn(),
        )

        # Main network
        if condition_method == "film":
            self._build_film_network(x_dim, hidden_dims, act_fn)
        else:
            self._build_concat_network(x_dim, hidden_dims, act_fn, cond_dim)

    def _build_film_network(self, x_dim, hidden_dims, act_fn):
        self.input_proj = nn.Linear(x_dim, hidden_dims[0])
        self.blocks = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.blocks.append(
                nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i + 1]), act_fn())
            )
            self.film_layers.append(FiLMLayer(hidden_dims[i + 1], hidden_dims[0]))
        self.output_proj = nn.Linear(hidden_dims[-1], x_dim)

    def _build_concat_network(self, x_dim, hidden_dims, act_fn, cond_dim):
        # TODO: Build concatenation-based conditioning
        raise NotImplementedError

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Predict velocity field v_θ(x_t, t, y).

        Args:
            x: Current state x_t, shape (batch, x_dim).
            t: Time, shape (batch,) or (batch, 1).
            y: Condition (measurement signal), shape (batch, y_dim).

        Returns:
            Predicted velocity, shape (batch, x_dim).
        """
        # Time embedding + condition encoding
        t_emb = self.time_embed(t)
        cond = self.cond_encoder(torch.cat([t_emb, y], dim=-1))

        if self.condition_method == "film":
            h = self.input_proj(x)
            for block, film in zip(self.blocks, self.film_layers):
                h = block(h)
                h = film(h, cond)
            return self.output_proj(h)
        else:
            raise NotImplementedError("Concat conditioning not yet implemented")
