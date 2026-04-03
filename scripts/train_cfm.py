"""Train Conditional Flow Matching model for any thermal inverse problem.

The same script works for Flash, 3-omega, TDTR, or any future method —
just change the config file.

Usage:
    python scripts/train_cfm.py --config configs/flash.yaml
    python scripts/train_cfm.py --config configs/three_omega.yaml
"""

import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from thermal_flow.forward import get_forward_model
from thermal_flow.models import VelocityNet, ConditionalFlowMatching, PhysicsConstrainedLoss
from thermal_flow.data import ThermalInverseDataset
from thermal_flow.utils import load_config, setup_wandb, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train CFM model")
    parser.add_argument("--config", type=str, default="configs/flash.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = torch.device(cfg.get("device", "cuda"))
    fm_cfg = cfg.forward_model
    model_cfg = cfg.model
    train_cfg = cfg.training

    # Initialize forward model (for spec + physics loss)
    fm_kwargs = {k: v for k, v in fm_cfg.items() if k != "name"}
    forward_model = get_forward_model(fm_cfg.name, **fm_kwargs).to(device)
    spec = forward_model.spec
    logger.info(f"Method: {spec.name} | θ_dim={spec.theta_dim}, y_dim={spec.y_dim}")

    # Load dataset
    data_dir = Path(cfg.dataset.save_dir)
    train_ds = ThermalInverseDataset(data_dir, split="train")
    val_ds = ThermalInverseDataset(data_dir, split="val")
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size)

    # Initialize model — auto-configured from forward model spec
    velocity_net = VelocityNet.from_spec(
        spec,
        hidden_dims=model_cfg.velocity_net.get("hidden_dims"),
        time_embed_dim=model_cfg.velocity_net.get("time_embed_dim", 128),
        activation=model_cfg.velocity_net.get("activation", "silu"),
    ).to(device)
    cfm = ConditionalFlowMatching(velocity_net, sigma_min=model_cfg.flow.sigma_min).to(device)

    # Physics loss (optional)
    physics_loss_fn = None
    if model_cfg.physics_loss.get("enabled", False):
        physics_loss_fn = PhysicsConstrainedLoss(
            forward_model, weight=model_cfg.physics_loss.weight
        )

    # Optimizer
    optimizer = torch.optim.AdamW(
        cfm.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    # Initialize W&B
    run = setup_wandb(cfg, run_name=f"cfm-{spec.name}")

    logger.info(f"Training for {train_cfg.epochs} epochs...")

    # TODO: Training loop
    #   for epoch in range(train_cfg.epochs):
    #     for batch in train_loader:
    #       theta, y = batch['theta'].to(device), batch['y'].to(device)
    #       loss = cfm.compute_loss(theta, y)
    #       if physics_loss_fn:
    #         theta_pred = cfm.sample(y, n_steps=5, n_samples=1)
    #         loss += physics_loss_fn(theta_pred, y)
    #       optimizer.zero_grad(); loss.backward(); optimizer.step()
    #       wandb.log({"loss": loss.item()})

    logger.info("CFM training complete.")


if __name__ == "__main__":
    main()
