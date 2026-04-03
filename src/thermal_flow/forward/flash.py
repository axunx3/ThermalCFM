"""Flash method (laser flash analysis) forward model.

The simplest thermal measurement technique — the ideal starting point
for validating the CFM framework (Phase 1, Minimum Viable Proof).

Only 2 parameters to invert: thermal diffusivity α and heat loss
coefficient h. The forward model is the Parker (1961) analytical solution
for the rear-face temperature rise of a thin disc after a pulsed laser.

Forward model:
    T_rear(t) = T_max · [1 + 2 Σ_{n=1}^{∞} (-1)^n exp(-n²π²αt/L²) · K_n(h)]

where L is the sample thickness and K_n(h) accounts for radiative
heat losses (Cape-Lehman correction).

Reference:
    Parker et al., J. Appl. Phys. 32, 1679 (1961)
    Cape & Lehman, J. Appl. Phys. 34, 1909 (1963)
"""

import torch
import torch.nn as nn
import numpy as np

from .base import ForwardModel, ForwardModelSpec


class FlashModel(ForwardModel):
    """Flash method forward model (Parker formula with heat loss correction).

    Args:
        thickness: Sample thickness L (m).
        n_time: Number of time points in the transient.
        t_max: Maximum measurement time (s).
        n_terms: Number of Fourier series terms for convergence.
        noise_level: Relative noise standard deviation.
    """

    def __init__(
        self,
        thickness: float = 1e-3,
        n_time: int = 200,
        t_max: float = 1.0,
        n_terms: int = 50,
        noise_level: float = 0.02,
    ):
        super().__init__()
        self.thickness = thickness
        self.n_time = n_time
        self.n_terms = n_terms
        self.noise_level = noise_level

        # Time array (excluding t=0 to avoid singularity)
        t = torch.linspace(t_max / n_time, t_max, n_time, dtype=torch.float64)
        self.register_buffer("t", t)

        # Precompute n² for Fourier series
        ns = torch.arange(1, n_terms + 1, dtype=torch.float64)
        self.register_buffer("ns", ns)
        self.register_buffer("ns_sq", ns ** 2)
        self.register_buffer("signs", (-1.0) ** ns)

        # Physical bounds for prior: [α, h]
        # α: thermal diffusivity (m²/s), typical range 1e-7 to 1e-4
        # h: Biot number (dimensionless heat loss), range 0 to ~5
        lower = torch.tensor([1e-7, 0.0], dtype=torch.float64)
        upper = torch.tensor([1e-4, 5.0], dtype=torch.float64)
        self.register_buffer("_lower", lower)
        self.register_buffer("_upper", upper)

    @property
    def spec(self) -> ForwardModelSpec:
        return ForwardModelSpec(
            name="flash",
            theta_dim=2,
            y_dim=self.n_time,
            theta_names=["alpha", "h"],
            y_names=[f"T_rear(t_{i})" for i in range(self.n_time)],
            theta_bounds=(self._lower, self._upper),
            description="Flash method: 2-parameter inversion (α, h) from rear-face transient",
        )

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute rear-face temperature transient T(t).

        Args:
            theta: Parameters [α, h], shape (batch, 2).

        Returns:
            Normalized temperature transient, shape (batch, n_time).
        """
        alpha = theta[:, 0:1]  # (batch, 1)
        h = theta[:, 1:2]      # (batch, 1)

        L = self.thickness
        t = self.t.unsqueeze(0)          # (1, n_time)
        ns_sq = self.ns_sq.unsqueeze(0)  # (1, n_terms)
        signs = self.signs.unsqueeze(0)  # (1, n_terms)

        # Dimensionless time: Fo = α·t / L²
        fo = alpha.unsqueeze(-1) * t.unsqueeze(-1) / (L ** 2)  # (batch, n_time, 1)

        # Heat loss correction (simplified Cape-Lehman)
        # K_n(h) ≈ 1 / (1 + h·n²π²) for small h
        ns_sq_exp = ns_sq.unsqueeze(1)  # (1, 1, n_terms)
        k_n = 1.0 / (1.0 + h.unsqueeze(-1) * ns_sq_exp * np.pi ** 2)  # (batch, 1, n_terms)

        # Fourier series: Σ (-1)^n · exp(-n²π²·Fo) · K_n
        exp_terms = torch.exp(-ns_sq_exp * np.pi ** 2 * fo)  # (batch, n_time, n_terms)
        series = (signs.unsqueeze(1) * exp_terms * k_n).sum(dim=-1)  # (batch, n_time)

        # T(t) = 1 + 2·Σ (normalized so T_max → 1)
        temperature = 1.0 + 2.0 * series

        return temperature.float()

    def sample_prior(
        self, n: int, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Sample [α, h] from a log-uniform (α) × uniform (h) prior.

        α is log-uniform in [1e-7, 1e-4] since it spans 3 orders of magnitude.
        h is uniform in [0, 5].
        """
        log_alpha = torch.empty(n, 1, device=device, dtype=torch.float64).uniform_(
            np.log(1e-7), np.log(1e-4)
        )
        alpha = torch.exp(log_alpha)
        h = torch.empty(n, 1, device=device, dtype=torch.float64).uniform_(0.0, 5.0)
        return torch.cat([alpha, h], dim=-1)

    def add_noise(self, y: torch.Tensor) -> torch.Tensor:
        """Add relative Gaussian noise to the temperature transient."""
        noise = torch.randn_like(y) * self.noise_level * y.abs().clamp(min=1e-10)
        return y + noise

    def log_transform(self, theta: torch.Tensor) -> torch.Tensor:
        """Log-transform α (spans orders of magnitude), leave h linear."""
        log_theta = theta.clone()
        log_theta[:, 0] = torch.log(theta[:, 0])
        return log_theta

    def exp_transform(self, log_theta: torch.Tensor) -> torch.Tensor:
        """Inverse of log_transform."""
        theta = log_theta.clone()
        theta[:, 0] = torch.exp(log_theta[:, 0])
        return theta
