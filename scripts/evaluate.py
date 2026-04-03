"""Evaluate and compare all methods on any thermal inverse problem.

Computes point-estimate metrics (MAE, RMSE), UQ metrics (NLL, calibration),
and validates against physical resolution limits.

Usage:
    python scripts/evaluate.py --config configs/flash.yaml --checkpoint outputs/flash/best.pt
"""

import argparse
import torch
from pathlib import Path

from thermal_flow.forward import get_forward_model
from thermal_flow.models import VelocityNet, ConditionalFlowMatching
from thermal_flow.inference import ODESampler, PosteriorSampler
from thermal_flow.evaluation import InversionMetrics, ResolutionLimitValidator
from thermal_flow.data import ThermalInverseDataset
from thermal_flow.utils import load_config, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate inversion methods")
    parser.add_argument("--config", type=str, default="configs/flash.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = torch.device(cfg.get("device", "cuda"))
    fm_cfg = cfg.forward_model

    fm_kwargs = {k: v for k, v in fm_cfg.items() if k != "name"}
    forward_model = get_forward_model(fm_cfg.name, **fm_kwargs).to(device)
    spec = forward_model.spec

    logger.info(f"Evaluating {spec.name} method")

    # TODO: Load model, run evaluation
    # 1. Load trained CFM checkpoint
    # 2. Load test dataset
    # 3. Run posterior sampling
    # 4. Compute all metrics
    # 5. For depth-resolved methods: validate against Burgholzer limit
    # 6. For multi-param methods: validate parameter ordering
    # 7. Save results

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
