"""Rectified Flow: Reflow iterations for trajectory straightening.

Forward-model agnostic. After initial CFM training, Reflow generates
(x_0, x_1) pairs by running the learned ODE, then retrains to straighten
the trajectories. This enables fewer integration steps at inference.

The deep connection: Reflow is a data-driven generalization of Burgholzer's
Virtual Wave transform — both straighten curved diffusion trajectories
into linear paths.

Reference:
    Liu et al., "Flow Straight and Fast" (2023)
"""

import torch
from .flow_matching import ConditionalFlowMatching


class RectifiedFlow:
    """Reflow iterations for trajectory straightening.

    Args:
        cfm: Trained ConditionalFlowMatching model.
        n_steps: Number of ODE steps for trajectory generation.
    """

    def __init__(
        self,
        cfm: ConditionalFlowMatching,
        n_steps: int = 100,
    ):
        self.cfm = cfm
        self.n_steps = n_steps

    def generate_reflow_pairs(
        self,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate (x_0, x_1) pairs by running the learned ODE.

        Args:
            y: Conditioning signals, shape (batch, y_dim).

        Returns:
            Tuple of (x_0, x_1) tensors for reflow training.
        """
        batch_size = y.shape[0]
        device = y.device
        x_dim = self.cfm.velocity_net.output_proj.out_features

        # Sample starting noise
        x_0 = torch.randn(batch_size, x_dim, device=device)

        # Integrate ODE to get x_1
        x = x_0.clone()
        dt = 1.0 / self.n_steps
        with torch.no_grad():
            for step in range(self.n_steps):
                t = torch.full((batch_size,), step * dt, device=device)
                v = self.cfm.velocity_net(x, t, y)
                x = x + v * dt

        x_1 = x
        return x_0, x_1

    def distill_one_step(self, cfm: ConditionalFlowMatching) -> None:
        """Distill the multi-step model into a one-step predictor.

        After sufficient Reflow rounds, trajectories are straight enough
        that a single Euler step suffices.
        """
        # TODO: One-step distillation training
        raise NotImplementedError
