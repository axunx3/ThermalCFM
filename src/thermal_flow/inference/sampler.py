"""ODE integration samplers for flow matching inference.

Supports both simple Euler integration and adaptive solvers via torchdiffeq.
"""

import torch
from torchdiffeq import odeint


class ODESampler:
    """ODE-based sampler for flow matching models.

    Integrates the learned velocity field from t=0 (noise) to t=1 (target).

    Args:
        velocity_net: Trained velocity network v_θ(x_t, t, y).
        solver: ODE solver ('euler', 'dopri5', 'rk4').
        n_steps: Number of integration steps (for fixed-step solvers).
        atol: Absolute tolerance (for adaptive solvers).
        rtol: Relative tolerance (for adaptive solvers).
    """

    def __init__(
        self,
        velocity_net,
        solver: str = "dopri5",
        n_steps: int = 20,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):
        self.velocity_net = velocity_net
        self.solver = solver
        self.n_steps = n_steps
        self.atol = atol
        self.rtol = rtol

    @torch.no_grad()
    def sample(
        self, y: torch.Tensor, x_0: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Generate a sample by integrating the ODE.

        Args:
            y: Conditioning signal, shape (batch, y_dim).
            x_0: Initial noise. If None, sampled from N(0,I).

        Returns:
            Generated κ profile at t=1, shape (batch, x_dim).
        """
        # TODO: Implement ODE integration
        raise NotImplementedError
