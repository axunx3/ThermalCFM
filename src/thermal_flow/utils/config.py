"""Configuration loading utilities."""

from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load YAML configuration with optional CLI overrides.

    Args:
        config_path: Path to the YAML config file.
        overrides: List of dotlist overrides, e.g. ["training.lr=1e-3"].

    Returns:
        Merged configuration as a DictConfig.
    """
    cfg = OmegaConf.load(config_path)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg
