"""Train the Conditional Flow Matching model.

Usage:
    python scripts/train_cfm.py [--config configs/cfm.yaml]
"""

import argparse
import torch
from torch.utils.data import DataLoader
from thermal_flow.models import VelocityNet, ConditionalFlowMatching, PhysicsConstrainedLoss
from thermal_flow.data import Thermal3OmegaDataset
from thermal_flow.utils import load_config, setup_wandb, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train CFM model")
    parser.add_argument("--config", type=str, default="configs/cfm.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = torch.device(cfg.get("device", "cuda"))

    # TODO: Initialize W&B
    # TODO: Load dataset and create DataLoader
    # TODO: Initialize VelocityNet and CFM
    # TODO: Training loop:
    #   - Sample batch
    #   - Compute CFM loss + optional physics loss
    #   - Optimizer step with gradient clipping
    #   - EMA update
    #   - Log to W&B
    # TODO: Save checkpoints

    logger.info("CFM training complete.")


if __name__ == "__main__":
    main()
