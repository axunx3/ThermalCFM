"""Velocity field network for Conditional Flow Matching.

Architecture: MLP with sinusoidal time embedding and FiLM conditioning
on the measurement signal. Automatically adapts to any forward model's
theta_dim and y_dim via the ForwardModelSpec.
"""

import math
import torch
import torch.nn as nn

from thermal_flow.forward.base import ForwardModelSpec


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for the time variable t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: output = gamma(cond) * x + beta(cond)."""

    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()
        self.scale = nn.Linear(condition_dim, feature_dim)
        self.shift = nn.Linear(condition_dim, feature_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.scale(condition) * x + self.shift(condition)


class VelocityNet(nn.Module):
    """MLP velocity field v_theta(x_t, t, y) for Conditional Flow Matching.

    Predicts the velocity field that transports noise x_0 ~ N(0,I) to
    the target theta distribution, conditioned on measurement y.

    Can be constructed either with explicit dimensions or from a ForwardModelSpec.

    Args:
        x_dim: Dimension of the state x (= theta_dim).
        y_dim: Dimension of the condition y (= y_dim from forward model).
        hidden_dims: List of hidden layer dimensions.
        time_embed_dim: Dimension of the sinusoidal time embedding.
        activation: Activation function name.
    """

    def __init__(
        self,
        x_dim: int = 100,
        y_dim: int = 120,
        hidden_dims: list[int] | None = None,
        time_embed_dim: int = 128,
        activation: str = "silu",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512, 512]

        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        act_fn = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}[activation]

        # Condition encoder: [time_emb, y] → hidden
        cond_dim = time_embed_dim + y_dim
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, hidden_dims[0]),
            act_fn(),
        )

        # FiLM-modulated main network
        self.input_proj = nn.Linear(x_dim, hidden_dims[0])
        self.blocks = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.blocks.append(
                nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i + 1]), act_fn())
            )
            self.film_layers.append(FiLMLayer(hidden_dims[i + 1], hidden_dims[0]))
        self.output_proj = nn.Linear(hidden_dims[-1], x_dim)

    @classmethod
    def from_spec(
        cls,
        spec: ForwardModelSpec,
        hidden_dims: list[int] | None = None,
        time_embed_dim: int = 128,
        activation: str = "silu",
    ) -> "VelocityNet":
        """Create a VelocityNet from a ForwardModelSpec.

        Automatically sets x_dim = theta_dim and y_dim = y_dim.
        Also auto-scales hidden dimensions for small problems.
        """
        if hidden_dims is None:
            # Scale network size with problem dimension
            if spec.theta_dim <= 10:
                hidden_dims = [256, 256, 256]
            elif spec.theta_dim <= 50:
                hidden_dims = [512, 512, 512, 512]
            else:
                hidden_dims = [512, 512, 512, 512, 512]

        return cls(
            x_dim=spec.theta_dim,
            y_dim=spec.y_dim,
            hidden_dims=hidden_dims,
            time_embed_dim=time_embed_dim,
            activation=activation,
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Predict velocity field v_theta(x_t, t, y).

        Args:
            x: Current state x_t, shape (batch, x_dim).
            t: Time, shape (batch,) or (batch, 1).
            y: Condition (measurement signal), shape (batch, y_dim).

        Returns:
            Predicted velocity, shape (batch, x_dim).
        """
        t_emb = self.time_embed(t)
        cond = self.cond_encoder(torch.cat([t_emb, y], dim=-1))

        h = self.input_proj(x)
        for block, film in zip(self.blocks, self.film_layers):
            h = block(h)
            h = film(h, cond)
        return self.output_proj(h)
