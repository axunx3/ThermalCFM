"""Generic PyTorch dataset for thermal inverse problems.

Works with any ForwardModel: the dataset stores (θ, y) pairs where θ is
the parameter vector and y is the noisy measurement. The forward model
defines the dimensions, prior distribution, and noise model.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Any

from thermal_flow.forward.base import ForwardModel


class ThermalInverseDataset(Dataset):
    """Dataset of (parameter, measurement) pairs for any thermal inverse problem.

    Each sample consists of:
        - theta: parameter vector θ, shape (theta_dim,)
        - y: noisy measurement, shape (y_dim,)

    Can be created in two ways:
        1. From a saved .pt file (for reproducible experiments)
        2. On-the-fly from a ForwardModel (for quick prototyping)

    Args:
        data_dir: Path to directory containing saved .pt files.
        split: One of 'train', 'val', 'test'.
        forward_model: If provided, used for log_transform.
        use_log_theta: Whether to store θ in log-space.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        split: str = "train",
        forward_model: ForwardModel | None = None,
        use_log_theta: bool = True,
    ):
        self.split = split
        self.forward_model = forward_model
        self.use_log_theta = use_log_theta

        if data_dir is not None:
            data = torch.load(Path(data_dir) / f"{split}.pt", weights_only=True)
            self.theta = data["theta"]
            self.y = data["y"]
        else:
            self.theta = None
            self.y = None

    @classmethod
    def from_forward_model(
        cls,
        forward_model: ForwardModel,
        n_samples: int,
        use_log_theta: bool = True,
        device: torch.device = torch.device("cpu"),
    ) -> "ThermalInverseDataset":
        """Generate dataset directly from a forward model.

        Args:
            forward_model: The forward model to use.
            n_samples: Number of (θ, y) pairs to generate.
            use_log_theta: Whether to store θ in log-space.
            device: Device for generation.

        Returns:
            ThermalInverseDataset instance.
        """
        dataset = cls(forward_model=forward_model, use_log_theta=use_log_theta)
        data = forward_model.generate_dataset(n_samples, device)

        theta = data["theta"].cpu().float()
        if use_log_theta:
            theta = forward_model.log_transform(theta)

        dataset.theta = theta
        dataset.y = data["y_noisy"].cpu().float()
        return dataset

    def __len__(self) -> int:
        if self.theta is None:
            return 0
        return self.theta.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single sample.

        Returns:
            Dict with keys 'theta' (parameter) and 'y' (measurement).
        """
        return {
            "theta": self.theta[idx],
            "y": self.y[idx],
        }

    def save(self, data_dir: str | Path, split: str | None = None) -> None:
        """Save dataset to disk.

        Args:
            data_dir: Directory to save to.
            split: Override split name.
        """
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        split = split or self.split
        torch.save({"theta": self.theta, "y": self.y}, data_dir / f"{split}.pt")

    def get_normalization_stats(self) -> dict[str, torch.Tensor]:
        """Compute mean and std for theta and y (for z-score normalization).

        Returns:
            Dict with 'theta_mean', 'theta_std', 'y_mean', 'y_std'.
        """
        return {
            "theta_mean": self.theta.mean(dim=0),
            "theta_std": self.theta.std(dim=0),
            "y_mean": self.y.mean(dim=0),
            "y_std": self.y.std(dim=0),
        }
