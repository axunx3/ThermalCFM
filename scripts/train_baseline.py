"""Train baseline methods (Feldman, Tikhonov, KRR, MLP).

Usage:
    python scripts/train_baseline.py [--config configs/baseline.yaml]
"""

import argparse
import torch
from thermal_flow.utils import load_config, setup_wandb, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train baseline methods")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)

    # TODO: Load dataset
    # TODO: Train each enabled baseline
    # TODO: Evaluate and log results to W&B

    logger.info("Baseline training complete.")


if __name__ == "__main__":
    main()
