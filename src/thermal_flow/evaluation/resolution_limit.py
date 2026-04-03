"""Validation against Burgholzer's thermodynamic resolution limit.

The theoretical resolution limit for thermal diffusion imaging is:
    Δz = 2π·z / ln(SNR)

This represents a fundamental thermodynamic bound: information recovery
from diffusion fields has exponentially increasing marginal cost with depth.

This validator checks whether the CFM posterior width is consistent with
this physical limit — a key contribution of the paper showing that CFM
learns physically meaningful uncertainty.

Reference:
    Burgholzer et al. (2017), "Virtual Wave Concept"
"""

import torch
import numpy as np


class ResolutionLimitValidator:
    """Validate CFM posterior against the Burgholzer resolution limit.

    Tests whether the learned model's uncertainty respects the fundamental
    resolution limit Δz = 2π·z / ln(SNR).

    Applicable to depth-resolved methods (3-omega, IR thermography).
    For multi-parameter methods (TDTR, Flash), validates that
    less-constrained parameters have wider posteriors.

    Args:
        z_grid: Depth grid points, shape (n_layers,). None for non-depth methods.
    """

    def __init__(self, z_grid: torch.Tensor | None = None):
        self.z_grid = z_grid

    def theoretical_limit(self, z: float, snr: float) -> float:
        """Compute the theoretical resolution limit at depth z.

        Args:
            z: Depth (m).
            snr: Signal-to-noise ratio (linear, not dB).

        Returns:
            Minimum resolvable feature size Δz (m).
        """
        if snr <= 1:
            return float("inf")
        return 2 * np.pi * z / np.log(snr)

    def validate_depth_resolved(
        self,
        posterior_std: torch.Tensor,
        snr: float,
    ) -> dict[str, np.ndarray]:
        """Check if posterior width follows the resolution limit for depth profiles.

        Args:
            posterior_std: Posterior standard deviation per depth,
                           shape (n_layers,) or (batch, n_layers) → averaged.
            snr: Measurement SNR.

        Returns:
            Dict with:
                - 'z': depth array (m)
                - 'posterior_std': observed posterior width
                - 'theoretical_limit': Δz(z) from Burgholzer
                - 'ratio': posterior_std / theoretical_limit
        """
        if self.z_grid is None:
            raise ValueError("z_grid required for depth-resolved validation")

        if posterior_std.dim() > 1:
            posterior_std = posterior_std.mean(dim=0)

        z = self.z_grid.numpy()
        theory = np.array([self.theoretical_limit(zi, snr) for zi in z])
        std = posterior_std.numpy()

        return {
            "z": z,
            "posterior_std": std,
            "theoretical_limit": theory,
            "ratio": std / np.where(theory > 0, theory, 1e-20),
        }

    def validate_parameter_ordering(
        self,
        posterior_std: torch.Tensor,
        param_names: list[str],
        sensitivity_ranking: list[str],
    ) -> dict:
        """For multi-parameter methods: check that less-sensitive parameters
        have wider posteriors.

        Args:
            posterior_std: Posterior std per parameter, shape (n_params,).
            param_names: Parameter names.
            sensitivity_ranking: Parameters ordered from most to least sensitive.

        Returns:
            Dict with ordering consistency check.
        """
        ranking_indices = [param_names.index(p) for p in sensitivity_ranking]
        stds = [posterior_std[i].item() for i in ranking_indices]
        is_monotonic = all(stds[i] <= stds[i + 1] for i in range(len(stds) - 1))

        return {
            "param_names": sensitivity_ranking,
            "posterior_stds": stds,
            "ordering_consistent": is_monotonic,
        }
