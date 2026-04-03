"""Physics-constrained loss using a differentiable forward model.

Works with any ForwardModel subclass. Adds a forward model consistency
term to the CFM loss:

    L_physics = || F(θ_pred) - y ||²

where F is the forward model and y is the measurement. Since all thermal
forward models in this framework are differentiable, gradients flow through
the physics constraint back to the velocity network.
"""

import torch
import torch.nn as nn

from thermal_flow.forward.base import ForwardModel


class PhysicsConstrainedLoss(nn.Module):
    """Forward model consistency loss for physics regularization.

    This is forward-model agnostic: it takes any ForwardModel subclass and
    uses it to verify that predicted θ produces measurements consistent
    with the observed data.

    Args:
        forward_model: Any differentiable ForwardModel instance.
        weight: Weight λ for the physics loss term.
    """

    def __init__(self, forward_model: ForwardModel, weight: float = 0.1):
        super().__init__()
        self.forward_model = forward_model
        self.weight = weight

    def forward(
        self,
        theta_pred: torch.Tensor,
        y_measured: torch.Tensor,
    ) -> torch.Tensor:
        """Compute physics consistency loss.

        If θ is in log-space, it is first transformed back via exp_transform
        before being passed through the forward model.

        Args:
            theta_pred: Predicted θ (possibly in log-space), shape (batch, theta_dim).
            y_measured: Measured signal, shape (batch, y_dim).

        Returns:
            Weighted physics loss scalar.
        """
        # Transform from log-space if needed
        theta_physical = self.forward_model.exp_transform(theta_pred)

        # Run forward model
        y_pred = self.forward_model(theta_physical)

        # L2 residual
        residual = nn.functional.mse_loss(y_pred, y_measured)
        return self.weight * residual
