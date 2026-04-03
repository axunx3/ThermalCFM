"""Evaluation metrics for thermal property inversion.

Forward-model agnostic. Includes both point-estimate metrics (MAE, RMSE)
and probabilistic metrics (NLL, calibration, sharpness) for UQ assessment.
All metrics work on generic (theta_pred, theta_true) pairs.
"""

import torch
import numpy as np

from thermal_flow.forward.base import ForwardModel


class InversionMetrics:
    """Compute evaluation metrics for thermal inversion results.

    All methods are static and work with any parameter dimensionality.
    """

    @staticmethod
    def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean Absolute Error."""
        return (pred - target).abs().mean()

    @staticmethod
    def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Root Mean Squared Error."""
        return ((pred - target) ** 2).mean().sqrt()

    @staticmethod
    def relative_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean relative error (percentage)."""
        return ((pred - target).abs() / target.abs().clamp(min=1e-10)).mean() * 100

    @staticmethod
    def per_param_rmse(
        pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """RMSE per parameter dimension.

        For depth profiles: gives depth-resolved RMSE.
        For multi-parameter: gives per-parameter RMSE.

        Args:
            pred: shape (batch, theta_dim).
            target: shape (batch, theta_dim).

        Returns:
            RMSE per dimension, shape (theta_dim,).
        """
        return ((pred - target) ** 2).mean(dim=0).sqrt()

    @staticmethod
    def forward_residual(
        pred_theta: torch.Tensor,
        y_measured: torch.Tensor,
        forward_model: ForwardModel,
    ) -> torch.Tensor:
        """Residual between forward-modeled signal and measurement.

        Args:
            pred_theta: Predicted θ (physical space).
            y_measured: Measured signal.
            forward_model: Forward model instance.

        Returns:
            L2 norm of signal residual.
        """
        with torch.no_grad():
            y_pred = forward_model(pred_theta)
        return ((y_pred - y_measured) ** 2).mean().sqrt()

    @staticmethod
    def calibration_curve(
        samples: torch.Tensor, target: torch.Tensor, n_bins: int = 20
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute calibration curve for posterior samples.

        For each confidence level p, checks what fraction of true values
        fall within the p-confidence interval. A well-calibrated model
        should give observed_coverage ≈ expected_coverage.

        Args:
            samples: Posterior samples, shape (batch, n_samples, theta_dim).
            target: True values, shape (batch, theta_dim).
            n_bins: Number of confidence levels.

        Returns:
            (expected_coverage, observed_coverage) arrays of shape (n_bins,).
        """
        expected = np.linspace(0, 1, n_bins + 2)[1:-1]
        observed = np.zeros_like(expected)

        for i, p in enumerate(expected):
            lower = samples.quantile((1 - p) / 2, dim=1)
            upper = samples.quantile((1 + p) / 2, dim=1)
            in_interval = ((target >= lower) & (target <= upper)).float()
            observed[i] = in_interval.mean().item()

        return expected, observed

    @staticmethod
    def nll(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood under Gaussian approximation."""
        mean = samples.mean(dim=1)
        std = samples.std(dim=1) + 1e-8
        nll = 0.5 * (
            torch.log(2 * torch.pi * std**2)
            + ((target - mean) / std) ** 2
        )
        return nll.mean()

    @staticmethod
    def sharpness(samples: torch.Tensor) -> torch.Tensor:
        """Average posterior standard deviation (lower = sharper)."""
        return samples.std(dim=1).mean()
