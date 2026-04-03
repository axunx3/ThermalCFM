"""Thermal conductivity profile generators for synthetic data.

Generates diverse κ(z) profiles using basis function superposition:
piecewise constant, exponential gradients, Gaussian bumps, and random
combinations. All profiles are bounded within physical limits.
"""

import torch
import numpy as np
from typing import Literal


class ProfileGenerator:
    """Generate random thermal conductivity depth profiles κ(z).

    Args:
        n_layers: Number of depth discretization points.
        z_max: Maximum depth (m).
        kappa_min: Minimum thermal conductivity (W/m·K).
        kappa_max: Maximum thermal conductivity (W/m·K).
    """

    def __init__(
        self,
        n_layers: int = 100,
        z_max: float = 50e-6,
        kappa_min: float = 0.1,
        kappa_max: float = 400.0,
    ):
        self.n_layers = n_layers
        self.z_max = z_max
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.z_grid = np.linspace(0, z_max, n_layers)

    def generate(
        self,
        n_samples: int,
        profile_type: Literal[
            "piecewise_constant", "exponential", "gaussian_bump", "random_combination"
        ] = "random_combination",
        n_basis_max: int = 5,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Generate a batch of κ(z) profiles.

        Args:
            n_samples: Number of profiles to generate.
            profile_type: Type of basis function.
            n_basis_max: Maximum number of basis functions per profile.
            seed: Random seed.

        Returns:
            Tensor of shape (n_samples, n_layers) with κ values in W/(m·K).
        """
        # TODO: Implement profile generation for each type
        raise NotImplementedError

    def _piecewise_constant(self, rng: np.random.Generator) -> np.ndarray:
        """Generate a piecewise constant κ(z) profile."""
        raise NotImplementedError

    def _exponential(self, rng: np.random.Generator) -> np.ndarray:
        """Generate an exponential gradient κ(z) profile."""
        raise NotImplementedError

    def _gaussian_bump(self, rng: np.random.Generator) -> np.ndarray:
        """Generate a Gaussian bump κ(z) profile."""
        raise NotImplementedError

    def _random_combination(
        self, rng: np.random.Generator, n_basis_max: int
    ) -> np.ndarray:
        """Generate a random superposition of basis functions."""
        raise NotImplementedError
