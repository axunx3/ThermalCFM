"""Generate synthetic 3-omega dataset for training and evaluation.

Usage:
    python scripts/generate_dataset.py [--config configs/dataset.yaml]
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm

from thermal_flow.forward import BorcaTasciucModel, ProfileGenerator, NoiseModel
from thermal_flow.utils import load_config, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic 3-omega dataset")
    parser.add_argument("--config", type=str, default="configs/dataset.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    logger.info(f"Generating dataset with config: {args.config}")

    # Initialize components
    # TODO: Set up forward model, profile generator, noise model
    # TODO: Generate (kappa, signal) pairs in batches
    # TODO: Split into train/val/test
    # TODO: Save to data directory

    logger.info("Dataset generation complete.")


if __name__ == "__main__":
    main()
