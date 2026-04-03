"""Uncertainty quantification via posterior sampling.

Forward-model agnostic. Generates multiple samples from the learned
posterior p(θ|y) by running the ODE with different initial noise
realizations. The spread of samples quantifies uncertainty for each
parameter dimension.

This is the core advantage of CFM over point-estimate methods (MLP, KRR):
    - Shallow/well-constrained parameters → narrow posterior → low uncertainty
    - Deep/poorly-constrained parameters → wide posterior → high uncertainty
    - Naturally aligns with Burgholzer's resolution limit Δz = 2πz/ln(SNR)
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
                - 'samples': shape (batch, n_samples, theta_dim)
                - 'mean': posterior mean, shape (batch, theta_dim)
                - 'std': posterior std, shape (batch, theta_dim)
                - 'median': posterior median, shape (batch, theta_dim)
                - 'q05': 5th percentile, shape (batch, theta_dim)
                - 'q95': 95th percentile, shape (batch, theta_dim)
        """
        batch_size = y.shape[0]
        device = y.device
        x_dim = self.sampler.velocity_net.output_proj.out_features

        # Collect samples from different noise initializations
        all_samples = []
        for _ in range(self.n_samples):
            x_0 = torch.randn(batch_size, x_dim, device=device)
            sample = self.sampler.sample(y, x_0)
            all_samples.append(sample)

        samples = torch.stack(all_samples, dim=1)  # (batch, n_samples, theta_dim)

        return {
            "samples": samples,
            "mean": samples.mean(dim=1),
            "std": samples.std(dim=1),
            "median": samples.median(dim=1).values,
            "q05": samples.quantile(0.05, dim=1),
            "q95": samples.quantile(0.95, dim=1),
        }
