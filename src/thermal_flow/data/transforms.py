"""Data transforms for thermal conductivity profiles and signals."""

import torch


class LogKappaTransform:
    """Transform κ to log(κ) space.

    Working in log-space guarantees positive thermal conductivity
    and handles multi-order-of-magnitude variations (0.1 - 400 W/mK).
    """

    def __call__(self, kappa: torch.Tensor) -> torch.Tensor:
        return torch.log(kappa)

    def inverse(self, log_kappa: torch.Tensor) -> torch.Tensor:
        return torch.exp(log_kappa)


class ZScoreNormalize:
    """Z-score normalization with stored statistics.

    Args:
        mean: Pre-computed mean.
        std: Pre-computed standard deviation.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-8)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + 1e-8) + self.mean

    @classmethod
    def fit(cls, data: torch.Tensor) -> "ZScoreNormalize":
        """Compute statistics from training data.

        Args:
            data: Training data, shape (n_samples, dim).

        Returns:
            Fitted ZScoreNormalize instance.
        """
        return cls(mean=data.mean(dim=0), std=data.std(dim=0))
