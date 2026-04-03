"""Time-Domain Thermoreflectance (TDTR) forward model.

Phase 2 extension: multi-parameter inversion of κ, G (interface
conductance), and c_p from pump-probe transient thermoreflectance signals.

The forward model computes the ratio signal -V_in/V_out(τ) as a function
of pump-probe delay time τ, using the multilayer frequency-domain
analytical solution with Hankel transform.

Reference:
    Cahill, Rev. Sci. Instrum. 75, 5119 (2004)
    Schmidt et al., Rev. Sci. Instrum. 79, 114902 (2008)
"""

import torch
import numpy as np

from .base import ForwardModel, ForwardModelSpec


class TDTRModel(ForwardModel):
    """TDTR forward model for multi-parameter thermal property inversion.

    The parameter vector θ contains per-layer thermal properties:
    [κ_1, G_1, c_p1, κ_2, G_2, c_p2, ...] where G_i is the interface
    conductance between layer i and i+1.

    For a typical 3-layer system (metal transducer / sample / substrate):
    θ = [κ_sample, G_metal_sample, G_sample_substrate, c_p_sample]

    Args:
        n_params: Number of parameters to invert.
        param_names: Names of parameters.
        n_delay: Number of pump-probe delay time points.
        tau_min: Minimum delay time (s).
        tau_max: Maximum delay time (s).
        pump_radius: 1/e² pump beam radius (m).
        probe_radius: 1/e² probe beam radius (m).
        modulation_freq: Pump modulation frequency (Hz).
        noise_level: Relative noise standard deviation.
    """

    def __init__(
        self,
        n_params: int = 4,
        param_names: list[str] | None = None,
        n_delay: int = 100,
        tau_min: float = 100e-12,
        tau_max: float = 5e-9,
        pump_radius: float = 10e-6,
        probe_radius: float = 5e-6,
        modulation_freq: float = 10e6,
        noise_level: float = 0.03,
    ):
        super().__init__()
        self.n_params = n_params
        self.noise_level = noise_level

        if param_names is None:
            param_names = ["kappa_sample", "G_metal_sample", "G_sample_sub", "cp_sample"]
        self.param_names = param_names

        # Log-spaced delay times
        tau = torch.logspace(
            np.log10(tau_min), np.log10(tau_max), n_delay, dtype=torch.float64
        )
        self.register_buffer("tau", tau)
        self.n_delay = n_delay

        self.pump_radius = pump_radius
        self.probe_radius = probe_radius
        self.modulation_freq = modulation_freq

        # Physical bounds: [κ (W/mK), G (MW/m²K), G, cp (MJ/m³K)]
        lower = torch.tensor([0.1, 1e6, 1e6, 0.5e6], dtype=torch.float64)
        upper = torch.tensor([400.0, 500e6, 500e6, 5e6], dtype=torch.float64)
        self.register_buffer("_lower", lower)
        self.register_buffer("_upper", upper)

    @property
    def spec(self) -> ForwardModelSpec:
        return ForwardModelSpec(
            name="tdtr",
            theta_dim=self.n_params,
            y_dim=self.n_delay,
            theta_names=self.param_names,
            y_names=[f"-Vin/Vout(tau_{i})" for i in range(self.n_delay)],
            theta_bounds=(self._lower, self._upper),
            description="TDTR: multi-parameter inversion (κ, G, cp) from pump-probe transient",
        )

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute -V_in/V_out(τ) ratio signal.

        Args:
            theta: Parameters, shape (batch, n_params).

        Returns:
            Ratio signal, shape (batch, n_delay).
        """
        # TODO: Implement frequency-domain multilayer solution
        # 1. For each modulation frequency harmonic:
        #    - Build transfer matrix for each layer
        #    - Apply Hankel transform for radial heat spreading
        # 2. Compute V_in and V_out from lock-in demodulation
        # 3. Return ratio -V_in/V_out
        raise NotImplementedError("TDTR forward model not yet implemented")

    def sample_prior(
        self, n: int, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Sample from log-uniform prior over physical ranges."""
        log_lower = torch.log(self._lower).to(device)
        log_upper = torch.log(self._upper).to(device)
        log_theta = torch.empty(n, self.n_params, device=device, dtype=torch.float64)
        for i in range(self.n_params):
            log_theta[:, i].uniform_(log_lower[i].item(), log_upper[i].item())
        return torch.exp(log_theta)

    def add_noise(self, y: torch.Tensor) -> torch.Tensor:
        """Add relative Gaussian noise to ratio signal."""
        noise = torch.randn_like(y) * self.noise_level * y.abs().clamp(min=1e-10)
        return y + noise

    def log_transform(self, theta: torch.Tensor) -> torch.Tensor:
        """All TDTR parameters span orders of magnitude → full log transform."""
        return torch.log(theta)

    def exp_transform(self, log_theta: torch.Tensor) -> torch.Tensor:
        return torch.exp(log_theta)
