"""Kernel Ridge Regression baseline for thermal property inversion.

Reference:
    Xiang et al. (2023) - KRR for FDTR inversion
"""

from sklearn.kernel_ridge import KernelRidge
import numpy as np
import torch


class KRRInversion:
    """Kernel Ridge Regression for signal-to-κ mapping.

    Args:
        kernel: Kernel type ('rbf', 'laplacian', 'polynomial').
        alpha: Regularization strength.
        gamma: Kernel coefficient (None for auto).
    """

    def __init__(
        self,
        kernel: str = "rbf",
        alpha: float = 1e-3,
        gamma: float | None = None,
    ):
        self.model = KernelRidge(kernel=kernel, alpha=alpha, gamma=gamma)

    def fit(self, signals: np.ndarray, kappas: np.ndarray) -> None:
        """Fit KRR model on training data.

        Args:
            signals: Training signals, shape (n_samples, n_freq * 2).
            kappas: Training κ profiles, shape (n_samples, n_layers).
        """
        self.model.fit(signals, kappas)

    def predict(self, signals: np.ndarray) -> np.ndarray:
        """Predict κ profiles from signals.

        Args:
            signals: Input signals, shape (n_samples, n_freq * 2).

        Returns:
            Predicted κ profiles, shape (n_samples, n_layers).
        """
        return self.model.predict(signals)
