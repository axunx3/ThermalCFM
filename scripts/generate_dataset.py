"""Generate synthetic dataset for any thermal measurement method.

The unified pipeline: pick a forward model → generate (θ, y) pairs.

Usage:
    python scripts/generate_dataset.py --config configs/flash.yaml
    python scripts/generate_dataset.py --config configs/three_omega.yaml
    python scripts/generate_dataset.py --config configs/tdtr.yaml
"""

import argparse
import torch
from pathlib import Path

from thermal_flow.forward import get_forward_model
from thermal_flow.data import ThermalInverseDataset
from thermal_flow.utils import load_config, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset")
    parser.add_argument("--config", type=str, default="configs/flash.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    ds_cfg = cfg.dataset
    fm_cfg = cfg.forward_model

    logger.info(f"Forward model: {fm_cfg.name}")
    logger.info(f"Generating {ds_cfg.n_train} train + {ds_cfg.n_val} val + {ds_cfg.n_test} test samples")

    # Instantiate forward model
    fm_kwargs = {k: v for k, v in fm_cfg.items() if k != "name"}
    forward_model = get_forward_model(fm_cfg.name, **fm_kwargs)
    logger.info(f"  θ dim: {forward_model.spec.theta_dim}, y dim: {forward_model.spec.y_dim}")

    save_dir = Path(ds_cfg.save_dir)
    device = torch.device(cfg.get("device", "cpu"))

    # Generate splits
    for split, n in [("train", ds_cfg.n_train), ("val", ds_cfg.n_val), ("test", ds_cfg.n_test)]:
        logger.info(f"Generating {split} ({n} samples)...")
        dataset = ThermalInverseDataset.from_forward_model(
            forward_model, n, use_log_theta=ds_cfg.use_log_theta, device=device,
        )
        dataset.save(save_dir, split=split)
        logger.info(f"  Saved to {save_dir / f'{split}.pt'}")

    # Save normalization stats from training set
    train_ds = ThermalInverseDataset(save_dir, split="train")
    stats = train_ds.get_normalization_stats()
    torch.save(stats, save_dir / "norm_stats.pt")
    logger.info(f"Normalization stats saved to {save_dir / 'norm_stats.pt'}")

    logger.info("Dataset generation complete.")


if __name__ == "__main__":
    main()
