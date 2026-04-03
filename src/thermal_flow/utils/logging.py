"""Logging and experiment tracking utilities."""

import logging
from pathlib import Path
from omegaconf import DictConfig


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger.

    Args:
        name: Logger name (typically __name__).
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def setup_wandb(cfg: DictConfig, run_name: str | None = None):
    """Initialize Weights & Biases run.

    Args:
        cfg: Configuration dict containing wandb settings.
        run_name: Optional run name override.

    Returns:
        wandb.Run instance.
    """
    import wandb

    wandb_cfg = cfg.get("wandb", {})
    run = wandb.init(
        project=wandb_cfg.get("project", "thermal-flow"),
        entity=wandb_cfg.get("entity", None),
        name=run_name,
        config=dict(cfg),
        mode=wandb_cfg.get("mode", "online"),
    )
    return run
