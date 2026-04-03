"""Uncertainty quantification via posterior sampling.

Generates multiple samples from the learned posterior p(κ|y) by running
the ODE with different initial noise realizations. The spread of samples
quantifies epistemic uncertainty at each depth.
"""

import torch
from .sampler import ODESampler


class PosteriorSampler:
    """Generate posterior samples for uncertainty quantification.

    Args:
        sampler: ODE sampler instance.
        n_samples: Number of posterior samples to draw.
    """

    def __init__(self, sampler: ODESampler, n_samples: int = 64):
        self.sampler = sampler
        self.n_samples = n_samples

    @torch.no_grad()
    def sample_posterior(
        self, y: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Draw posterior samples and compute statistics.

        Args:
            y: Conditioning signal, shape (batch, y_dim).

        Returns:
            Dict with keys:
                - 'samples': shape (batch, n_samples, x_dim)
                - 'mean': posterior mean, shape (batch, x_dim)
                - 'std': posterior std, shape (batch, x_dim)
                - 'median': posterior median, shape (batch, x_dim)
                - 'q05': 5th percentile, shape (batch, x_dim)
                - 'q95': 95th percentile, shape (batch, x_dim)
        """
        # TODO: Multiple ODE integrations with different x_0
        raise NotImplementedError
