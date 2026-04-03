"""Evaluation metrics for thermal conductivity inversion.

Includes both point-estimate metrics (MAE, RMSE) and probabilistic
metrics (NLL, calibration, sharpness) for uncertainty quantification.
"""

import torch
import numpy as np


class InversionMetrics:
    """Compute evaluation metrics for κ(z) inversion results."""

    @staticmethod
    def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean Absolute Error."""
        return (pred - target).abs().mean()

    @staticmethod
    def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Root Mean Squared Error."""
        return ((pred - target) ** 2).mean().sqrt()

    @staticmethod
    def depth_resolved_rmse(
        pred: torch.Tensor, target: torch.Tensor, z_grid: torch.Tensor
    ) -> torch.Tensor:
        """RMSE at each depth point.

        Args:
            pred: Predicted κ, shape (batch, n_layers).
            target: True κ, shape (batch, n_layers).
            z_grid: Depth grid, shape (n_layers,).

        Returns:
            RMSE per depth, shape (n_layers,).
        """
        return ((pred - target) ** 2).mean(dim=0).sqrt()

    @staticmethod
    def forward_residual(
        pred_kappa: torch.Tensor,
        measured_signal: torch.Tensor,
        forward_model,
        **forward_kwargs,
    ) -> torch.Tensor:
        """Residual between forward-modeled signal and measurement.

        Args:
            pred_kappa: Predicted κ profile.
            measured_signal: Measured V_3ω signal.
            forward_model: Forward model instance.

        Returns:
            L2 norm of signal residual.
        """
        # TODO: Compute forward model on pred_kappa, compare
        raise NotImplementedError

    @staticmethod
    def calibration_curve(
        samples: torch.Tensor, target: torch.Tensor, n_bins: int = 20
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute calibration curve for posterior samples.

        For each confidence level p, checks what fraction of true values
        fall within the p-confidence interval.

        Args:
            samples: Posterior samples, shape (batch, n_samples, x_dim).
            target: True values, shape (batch, x_dim).
            n_bins: Number of confidence levels.

        Returns:
            (expected_coverage, observed_coverage) arrays.
        """
        # TODO: Implement calibration check
        raise NotImplementedError

    @staticmethod
    def nll(
        samples: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Negative log-likelihood under Gaussian approximation.

        Args:
            samples: Posterior samples, shape (batch, n_samples, x_dim).
            target: True values, shape (batch, x_dim).

        Returns:
            Average NLL scalar.
        """
        mean = samples.mean(dim=1)
        std = samples.std(dim=1) + 1e-8
        nll = 0.5 * (
            torch.log(2 * torch.pi * std**2)
            + ((target - mean) / std) ** 2
        )
        return nll.mean()
