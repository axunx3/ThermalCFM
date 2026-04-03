"""ODE integration samplers for flow matching inference.

Forward-model agnostic. Supports both simple Euler integration and
adaptive solvers via torchdiffeq.
"""

import torch
from torchdiffeq import odeint


class ODESampler:
    """ODE-based sampler for flow matching models.

    Integrates the learned velocity field from t=0 (noise) to t=1 (target).
    Works with any theta_dim / y_dim — determined at runtime from the
    velocity network.

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
            Generated θ at t=1, shape (batch, theta_dim).
        """
        device = y.device
        x_dim = self.velocity_net.output_proj.out_features
        batch_size = y.shape[0]

        if x_0 is None:
            x_0 = torch.randn(batch_size, x_dim, device=device)

        if self.solver == "euler":
            # Simple Euler integration
            x = x_0
            dt = 1.0 / self.n_steps
            for step in range(self.n_steps):
                t = torch.full((batch_size,), step * dt, device=device)
                v = self.velocity_net(x, t, y)
                x = x + v * dt
            return x
        else:
            # Adaptive solver via torchdiffeq
            def ode_fn(t_scalar, x):
                t_batch = torch.full((batch_size,), t_scalar.item(), device=device)
                return self.velocity_net(x, t_batch, y)

            t_span = torch.tensor([0.0, 1.0], device=device)
            solution = odeint(
                ode_fn, x_0, t_span,
                method=self.solver, atol=self.atol, rtol=self.rtol,
            )
            return solution[-1]  # x at t=1
