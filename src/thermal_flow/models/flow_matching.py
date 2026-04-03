"""Conditional Flow Matching training logic.

Forward-model agnostic: works with any (theta, y) pairs regardless of
which thermal measurement technique generated them. The CFM objective is:

    L(θ_net) = E_{t, x_0, x_1} || v_θ(x_t, t, y) - (x_1 - x_0) ||²

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

    This is the core of the universal thermal inversion framework.
    It operates purely on (theta, y) pairs and knows nothing about
    the specific measurement technique.

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
            x_1: Target samples (θ in whatever space), shape (batch, theta_dim).
            y: Conditioning signals (measurements), shape (batch, y_dim).

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

        For each measurement y, generates n_samples posterior samples
        by integrating the learned velocity field from t=0 to t=1.

        Args:
            y: Conditioning signal, shape (batch, y_dim).
            n_steps: Number of Euler integration steps.
            n_samples: Number of posterior samples per condition.

        Returns:
            Generated θ samples, shape (batch * n_samples, theta_dim).
        """
        batch_size = y.shape[0]
        device = y.device

        # Repeat y for multiple samples
        if n_samples > 1:
            y = y.repeat_interleave(n_samples, dim=0)

        # Start from noise
        x_dim = self.velocity_net.output_proj.out_features
        x = torch.randn(batch_size * n_samples, x_dim, device=device)

        # Euler integration
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = torch.full((x.shape[0],), step * dt, device=device)
            v = self.velocity_net(x, t, y)
            x = x + v * dt

        return x
