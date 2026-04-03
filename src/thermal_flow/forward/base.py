"""Abstract base class for thermal forward models.

All thermal measurement techniques share the same inverse problem structure:

    y = F[θ] + η

where θ is the thermal parameter(s) to invert, F is the forward model,
y is the measurement, and η is noise. This base class defines the
pluggable interface that the CFM framework operates on.

Any concrete forward model only needs to implement:
    1. forward(): θ → y (deterministic mapping)
    2. sample_prior(): draw θ from the prior distribution p(θ)
    3. add_noise(): inject realistic measurement noise

The CFM training loop, inference, and evaluation are completely agnostic
to which forward model is plugged in.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ForwardModelSpec:
    """Specification of a forward model's input/output dimensions and metadata.

    This is used by the CFM pipeline to automatically configure
    network architecture, dataset shape, and evaluation metrics.
    """

    name: str                  # e.g. "flash", "3omega", "tdtr"
    theta_dim: int             # dimension of parameter vector θ
    y_dim: int                 # dimension of measurement vector y
    theta_names: list[str]     # human-readable parameter names
    y_names: list[str]         # human-readable measurement names
    theta_bounds: tuple[torch.Tensor, torch.Tensor]  # (lower, upper) physical bounds
    description: str = ""


class ForwardModel(nn.Module, ABC):
    """Abstract base class for all thermal forward models.

    Subclasses must implement:
        - spec: property returning ForwardModelSpec
        - forward(theta) -> y: the deterministic forward mapping
        - sample_prior(n, device) -> theta: sample from prior p(θ)
        - add_noise(y) -> y_noisy: inject measurement noise

    Optional overrides:
        - log_transform(theta) / exp_transform(log_theta): if working in
          log-space is preferred (e.g. for conductivity)
    """

    @property
    @abstractmethod
    def spec(self) -> ForwardModelSpec:
        """Return the model specification."""
        ...

    @abstractmethod
    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute the forward model F[θ].

        Args:
            theta: Parameter vector, shape (batch, theta_dim).

        Returns:
            Simulated measurement, shape (batch, y_dim).
        """
        ...

    @abstractmethod
    def sample_prior(
        self, n: int, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Sample n parameter vectors from the prior distribution p(θ).

        The prior should cover the physically meaningful range of parameters
        while being broad enough for the CFM to learn the posterior well.

        Args:
            n: Number of samples.
            device: Target device.

        Returns:
            Parameter vectors, shape (n, theta_dim).
        """
        ...

    @abstractmethod
    def add_noise(self, y: torch.Tensor) -> torch.Tensor:
        """Add realistic measurement noise to clean signals.

        Args:
            y: Clean forward model output, shape (batch, y_dim).

        Returns:
            Noisy measurement, shape (batch, y_dim).
        """
        ...

    def log_transform(self, theta: torch.Tensor) -> torch.Tensor:
        """Transform θ to log-space (default: identity).

        Override this for parameters spanning multiple orders of magnitude
        (e.g. thermal conductivity 0.1–400 W/mK).
        """
        return theta

    def exp_transform(self, log_theta: torch.Tensor) -> torch.Tensor:
        """Inverse of log_transform (default: identity)."""
        return log_theta

    def generate_dataset(
        self, n: int, device: torch.device = torch.device("cpu")
    ) -> dict[str, torch.Tensor]:
        """Generate a synthetic (θ, y) dataset.

        This is the standard data generation pipeline for CFM training:
            1. Sample θ ~ p(θ)
            2. Compute y_clean = F[θ]
            3. Add noise: y = y_clean + η

        Args:
            n: Number of samples.
            device: Target device.

        Returns:
            Dict with keys 'theta', 'y_clean', 'y_noisy'.
        """
        theta = self.sample_prior(n, device)
        with torch.no_grad():
            y_clean = self.forward(theta)
        y_noisy = self.add_noise(y_clean)
        return {"theta": theta, "y_clean": y_clean, "y_noisy": y_noisy}
