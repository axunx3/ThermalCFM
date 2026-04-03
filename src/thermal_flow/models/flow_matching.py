"""Conditional Flow Matching training logic.

Implements the CFM objective:
    L(θ) = E_{t, x_0, x_1} || v_θ(x_t, t, y) - (x_1 - x_0) ||²

where x_t = (1-t)x_0 + t·x_1 is the linear interpolation between
noise x_0 and target x_1, and y is the measurement condition.

Reference:
    Lipman et al., "Flow Matching for Generative Modeling" (2023)
"""

import torch
import torch.nn as nn
from .velocity_net import VelocityNet


class ConditionalFlowMatching(nn.Module):
    """Conditional Flow Matching training wrapper.

    Args:
        velocity_net: The velocity field network v_θ.
        sigma_min: Minimum noise scale for the interpolation path.
    """

    def __init__(
        self,
        velocity_net: VelocityNet,
        sigma_min: float = 1e-4,
    ):
        super().__init__()
        self.velocity_net = velocity_net
        self.sigma_min = sigma_min

    def compute_loss(
        self,
        x_1: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the CFM training loss.

        Args:
            x_1: Target samples (κ profiles), shape (batch, x_dim).
            y: Conditioning signals (V_3ω), shape (batch, y_dim).

        Returns:
            Scalar loss value.
        """
        batch_size = x_1.shape[0]
        device = x_1.device

        # Sample time uniformly
        t = torch.rand(batch_size, device=device)

        # Sample noise
        x_0 = torch.randn_like(x_1)

        # Linear interpolation: x_t = (1-t)x_0 + t*x_1
        t_expanded = t[:, None]
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1

        # Target velocity: dx/dt = x_1 - x_0
        target_velocity = x_1 - x_0

        # Predicted velocity
        pred_velocity = self.velocity_net(x_t, t, y)

        # MSE loss
        loss = nn.functional.mse_loss(pred_velocity, target_velocity)
        return loss

    @torch.no_grad()
    def sample(
        self,
        y: torch.Tensor,
        n_steps: int = 20,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Generate posterior samples via Euler ODE integration.

        Args:
            y: Conditioning signal, shape (batch, y_dim).
            n_steps: Number of Euler integration steps.
            n_samples: Number of posterior samples per condition.

        Returns:
            Generated κ profiles, shape (batch * n_samples, x_dim).
        """
        # TODO: Implement ODE integration (Euler or use torchdiffeq)
        raise NotImplementedError
