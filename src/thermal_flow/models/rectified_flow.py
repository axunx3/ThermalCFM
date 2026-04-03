"""Rectified Flow: Reflow iterations for trajectory straightening.

After initial CFM training, Reflow generates (x_0, x_1) pairs by running
the learned ODE, then retrains to straighten the trajectories. This enables
fewer integration steps (or even one-step generation) at inference.

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
        n_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate (x_0, x_1) pairs by running the learned ODE.

        Starts from noise x_0 and integrates to get x_1, creating
        paired data for the next round of training.

        Args:
            y: Conditioning signals, shape (batch, y_dim).
            n_samples: Number of pairs to generate.

        Returns:
            Tuple of (x_0, x_1) tensors for reflow training.
        """
        # TODO: Run ODE forward to generate paired trajectories
        raise NotImplementedError

    def distill_one_step(self, cfm: ConditionalFlowMatching) -> None:
        """Distill the multi-step model into a one-step predictor.

        After sufficient Reflow rounds, trajectories are straight enough
        that a single Euler step suffices.

        Args:
            cfm: The reflowed CFM model to distill.
        """
        # TODO: One-step distillation training
        raise NotImplementedError
