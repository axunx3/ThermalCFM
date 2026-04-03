"""Run Reflow iterations for trajectory straightening.

Usage:
    python scripts/reflow.py --checkpoint outputs/cfm/best.pt [--config configs/cfm.yaml]
"""

import argparse
import torch
from thermal_flow.models import RectifiedFlow, ConditionalFlowMatching
from thermal_flow.utils import load_config, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Reflow iterations")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/cfm.yaml")
    parser.add_argument("--n-rounds", type=int, default=2)
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)

    # TODO: Load trained CFM model
    # TODO: For each reflow round:
    #   - Generate (x_0, x_1) pairs
    #   - Retrain CFM on straightened pairs
    # TODO: Optional one-step distillation

    logger.info("Reflow complete.")


if __name__ == "__main__":
    main()
