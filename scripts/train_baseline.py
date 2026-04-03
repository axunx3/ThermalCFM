"""Train baseline methods for comparison.

Baselines are forward-model agnostic: MLP and KRR work on any (θ, y) pairs.
Feldman and Tikhonov are specific to certain methods.

Usage:
    python scripts/train_baseline.py --config configs/flash.yaml
"""

import argparse
import torch
from pathlib import Path

from thermal_flow.forward import get_forward_model
from thermal_flow.data import ThermalInverseDataset
from thermal_flow.utils import load_config, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train baseline methods")
    parser.add_argument("--config", type=str, default="configs/flash.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    fm_cfg = cfg.forward_model

    fm_kwargs = {k: v for k, v in fm_cfg.items() if k != "name"}
    forward_model = get_forward_model(fm_cfg.name, **fm_kwargs)
    spec = forward_model.spec

    # Load dataset
    data_dir = Path(cfg.dataset.save_dir)
    train_ds = ThermalInverseDataset(data_dir, split="train")
    test_ds = ThermalInverseDataset(data_dir, split="test")

    logger.info(f"Method: {spec.name} | Training baselines...")

    # TODO: Train MLP baseline (works for any method)
    # TODO: Train KRR baseline (works for any method)
    # TODO: If 3omega: also run Feldman and Tikhonov
    # TODO: Evaluate and log results

    logger.info("Baseline training complete.")


if __name__ == "__main__":
    main()
