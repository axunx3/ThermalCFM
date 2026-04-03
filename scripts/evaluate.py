"""Evaluate and compare all methods.

Usage:
    python scripts/evaluate.py --checkpoint outputs/cfm/best.pt [--config configs/cfm.yaml]
"""

import argparse
import torch
from thermal_flow.evaluation import InversionMetrics, ResolutionLimitValidator
from thermal_flow.inference import PosteriorSampler, ODESampler
from thermal_flow.utils import load_config, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate inversion methods")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/cfm.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)

    # TODO: Load models (CFM + baselines)
    # TODO: Load test dataset
    # TODO: Run inference + posterior sampling
    # TODO: Compute all metrics (MAE, RMSE, depth-RMSE, NLL, calibration)
    # TODO: Validate against Burgholzer resolution limit
    # TODO: Save results and figures

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
