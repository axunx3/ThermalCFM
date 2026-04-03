"""Physics-constrained loss using the differentiable forward model.

Adds a forward model consistency term to the CFM loss:
    L_physics = || F(κ_pred) - y ||²

where F is the Borca-Tasciuc forward model and y is the measurement.
The forward model is fully differentiable, so gradients flow through both
the CFM velocity net and the physics constraint.
"""

import torch
import torch.nn as nn


class PhysicsConstrainedLoss(nn.Module):
    """Forward model consistency loss for physics regularization.

    Args:
        forward_model: Differentiable 3-omega forward model.
        weight: Weight λ for the physics loss term.
    """

    def __init__(self, forward_model, weight: float = 0.1):
        super().__init__()
        self.forward_model = forward_model
        self.weight = weight

    def forward(
        self,
        kappa_pred: torch.Tensor,
        signal_measured: torch.Tensor,
        rho_cp: torch.Tensor,
        layer_thickness: torch.Tensor,
    ) -> torch.Tensor:
        """Compute physics consistency loss.

        Args:
            kappa_pred: Predicted κ profile, shape (batch, n_layers).
            signal_measured: Measured V_3ω, shape (batch, n_freq * 2).
            rho_cp: Volumetric heat capacity, shape (batch, n_layers).
            layer_thickness: Layer thicknesses, shape (batch, n_layers).

        Returns:
            Weighted physics loss scalar.
        """
        # TODO: Run forward model on predicted κ, compare with measurement
        raise NotImplementedError
