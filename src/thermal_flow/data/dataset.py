"""PyTorch dataset for synthetic 3-omega (κ, V_3ω) pairs."""

import torch
from torch.utils.data import Dataset
from pathlib import Path


class Thermal3OmegaDataset(Dataset):
    """Dataset of (thermal conductivity profile, 3-omega signal) pairs.

    Each sample consists of:
        - kappa: κ(z) depth profile, shape (n_layers,)
        - signal: Complex V_3ω(f) measurement, shape (n_freq,) or (2*n_freq,)
          where real and imaginary parts are concatenated.

    Args:
        data_dir: Path to directory containing saved .pt files.
        split: One of 'train', 'val', 'test'.
        transform: Optional transform applied to (kappa, signal) pairs.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        # TODO: Load precomputed data
        self.kappa = None
        self.signal = None

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        """Return a single sample.

        Returns:
            Dict with keys 'kappa' and 'signal'.
        """
        raise NotImplementedError
