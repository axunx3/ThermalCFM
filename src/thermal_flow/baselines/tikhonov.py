"""Tikhonov regularized inversion for 3-omega depth profiling."""

import torch
import numpy as np


class TikhonovInversion:
    """Tikhonov regularization for the 3-omega inverse problem.

    Solves: min_κ ||A·κ - y||² + α||L·κ||²
    where A is the linearized forward operator, y is the measurement,
    and L is a smoothness-promoting regularization matrix.

    Args:
        alpha: Regularization parameter.
        regularization_order: Order of the difference matrix L (0, 1, or 2).
    """

    def __init__(self, alpha: float = 1e-3, regularization_order: int = 1):
        self.alpha = alpha
        self.regularization_order = regularization_order

    def solve(
        self, signal: torch.Tensor, forward_model, **kwargs
    ) -> torch.Tensor:
        """Solve the regularized inverse problem.

        Args:
            signal: Measured V_3ω, shape (batch, n_freq).
            forward_model: The forward model instance for Jacobian computation.

        Returns:
            Reconstructed κ(z), shape (batch, n_layers).
        """
        # TODO: Compute Jacobian, form normal equations, solve
        raise NotImplementedError
