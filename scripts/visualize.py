"""Visualization utilities for results and figures.

Usage:
    python scripts/visualize.py --results outputs/eval/results.pt
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_kappa_profile(
    z_grid: np.ndarray,
    kappa_true: np.ndarray,
    kappa_pred: np.ndarray,
    kappa_samples: np.ndarray | None = None,
    save_path: str | None = None,
):
    """Plot true vs predicted κ(z) with uncertainty bands.

    Args:
        z_grid: Depth array (m), shape (n_layers,).
        kappa_true: True κ profile, shape (n_layers,).
        kappa_pred: Mean predicted κ, shape (n_layers,).
        kappa_samples: Posterior samples, shape (n_samples, n_layers).
        save_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(6, 8))

    z_um = z_grid * 1e6  # convert to μm

    if kappa_samples is not None:
        q05 = np.percentile(kappa_samples, 5, axis=0)
        q95 = np.percentile(kappa_samples, 95, axis=0)
        ax.fill_betweenx(z_um, q05, q95, alpha=0.3, label="90% CI")

    ax.plot(kappa_true, z_um, "k-", linewidth=2, label="True")
    ax.plot(kappa_pred, z_um, "r--", linewidth=1.5, label="Predicted")

    ax.set_xlabel("κ (W/m·K)")
    ax.set_ylabel("Depth (μm)")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("Thermal Conductivity Depth Profile")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize results")
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="figures/")
    args = parser.parse_args()

    # TODO: Load results and generate paper figures

    print(f"Figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
