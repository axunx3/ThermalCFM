"""Borca-Tasciuc transfer matrix forward model for the 3-omega method.

Phase 2 of the research roadmap: depth-resolved κ(z) inversion from
frequency-sweep V_3ω measurements. ~100-dimensional inverse problem.

Implements the analytical solution for temperature rise at a heater line on
a multilayer substrate, following the transfer matrix formalism. Fully
differentiable via PyTorch autograd.

Reference:
    Borca-Tasciuc et al., Rev. Sci. Instrum. 72, 2139 (2001)
"""

import torch
import numpy as np

from .base import ForwardModel, ForwardModelSpec


class ThreeOmegaModel(ForwardModel):
    """3-omega forward model using the Borca-Tasciuc transfer matrix method.

    Given a thermal conductivity depth profile κ(z), computes the complex
    3-omega voltage signal V_3ω(f) at each measurement frequency.

    The parameter vector θ = κ(z) is the discretized thermal conductivity
    profile (n_layers values). The measurement vector y = [Re(V_3ω), Im(V_3ω)]
    is the concatenated real and imaginary parts of the voltage signal.

    Args:
        n_layers: Number of depth discretization layers.
        heater_half_width: Half-width of the heater line b (m).
        n_freq: Number of frequency points in the sweep.
        freq_min: Minimum frequency (Hz).
        freq_max: Maximum frequency (Hz).
        z_max: Maximum depth (m).
        rho_cp_default: Default volumetric heat capacity (J/m³·K).
        n_quadrature: Number of Gauss-Legendre quadrature points.
        k_max: Spatial frequency cutoff for Fourier integral.
        kappa_min: Minimum thermal conductivity for prior (W/m·K).
        kappa_max: Maximum thermal conductivity for prior (W/m·K).
        noise_real: Relative noise on real part.
        noise_imag: Relative noise on imaginary part.
    """

    def __init__(
        self,
        n_layers: int = 100,
        heater_half_width: float = 5e-6,
        n_freq: int = 60,
        freq_min: float = 10.0,
        freq_max: float = 1e5,
        z_max: float = 50e-6,
        rho_cp_default: float = 1.63e6,
        n_quadrature: int = 64,
        k_max: float = 1e7,
        kappa_min: float = 0.1,
        kappa_max: float = 400.0,
        noise_real: float = 0.03,
        noise_imag: float = 0.07,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.heater_half_width = heater_half_width
        self.n_freq = n_freq
        self.n_quadrature = n_quadrature
        self.z_max = z_max
        self.rho_cp_default = rho_cp_default
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.noise_real = noise_real
        self.noise_imag = noise_imag

        # Log-spaced frequency array
        freqs = torch.logspace(np.log10(freq_min), np.log10(freq_max), n_freq)
        self.register_buffer("freqs", freqs)
        self.register_buffer("omega", 2 * np.pi * freqs)

        # Uniform layer thickness
        layer_thickness = torch.full((n_layers,), z_max / n_layers, dtype=torch.float64)
        self.register_buffer("layer_thickness", layer_thickness)

        # Gauss-Legendre quadrature nodes and weights
        nodes, weights = np.polynomial.legendre.leggauss(n_quadrature)
        k_nodes = torch.tensor((nodes + 1) / 2 * k_max, dtype=torch.float64)
        k_weights = torch.tensor(weights * k_max / 2, dtype=torch.float64)
        self.register_buffer("k_nodes", k_nodes)
        self.register_buffer("k_weights", k_weights)

        # Physical bounds
        lower = torch.full((n_layers,), kappa_min, dtype=torch.float64)
        upper = torch.full((n_layers,), kappa_max, dtype=torch.float64)
        self.register_buffer("_lower", lower)
        self.register_buffer("_upper", upper)

    @property
    def spec(self) -> ForwardModelSpec:
        return ForwardModelSpec(
            name="3omega",
            theta_dim=self.n_layers,
            y_dim=self.n_freq * 2,  # real + imaginary
            theta_names=[f"kappa_layer_{i}" for i in range(self.n_layers)],
            y_names=(
                [f"Re(V3w)_f{i}" for i in range(self.n_freq)]
                + [f"Im(V3w)_f{i}" for i in range(self.n_freq)]
            ),
            theta_bounds=(self._lower, self._upper),
            description="3-omega: depth-resolved κ(z) from V_3ω frequency sweep",
        )

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute 3-omega signal from κ(z) profile.

        Args:
            theta: κ per layer, shape (batch, n_layers).

        Returns:
            [Re(V_3ω), Im(V_3ω)] concatenated, shape (batch, 2*n_freq).
        """
        # TODO: Implement transfer matrix computation
        # 1. For each frequency ω and spatial frequency k:
        #    q_i = sqrt(k² · κ_x/κ_y + j·2ω·ρ·cp/κ_y)
        # 2. Recursive impedance from substrate to surface
        # 3. Fourier integral with heater width weighting: sin(kb)/(kb)
        # 4. Gauss-Legendre quadrature over k
        # 5. Concatenate real and imaginary parts
        raise NotImplementedError("Transfer matrix computation not yet implemented")

    def sample_prior(
        self, n: int, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Sample κ(z) profiles from a log-uniform prior.

        Generates random profiles using basis function superposition:
        piecewise constant, exponential, Gaussian bump, and combinations.
        """
        # Log-uniform sampling across the physical range
        log_kappa = torch.empty(n, self.n_layers, device=device, dtype=torch.float64)
        log_kappa.uniform_(np.log(self.kappa_min), np.log(self.kappa_max))
        # TODO: Replace with structured basis function generation (profiles.py)
        return torch.exp(log_kappa)

    def add_noise(self, y: torch.Tensor) -> torch.Tensor:
        """Add noise with different levels for real and imaginary parts."""
        n_freq = self.n_freq
        noise = torch.randn_like(y)
        # Real part: lower noise
        noise[:, :n_freq] *= self.noise_real
        # Imaginary part: higher noise
        noise[:, n_freq:] *= self.noise_imag
        return y + noise * y.abs().clamp(min=1e-10)

    def log_transform(self, theta: torch.Tensor) -> torch.Tensor:
        """Log-transform κ (spans 0.1–400 W/mK)."""
        return torch.log(theta)

    def exp_transform(self, log_theta: torch.Tensor) -> torch.Tensor:
        return torch.exp(log_theta)
