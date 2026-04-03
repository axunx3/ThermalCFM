# Thermal Flow

**Conditional Flow Matching for 3-omega Thermal Conductivity Depth Profiling**

Solving the ill-posed inverse problem of reconstructing thermal conductivity depth profiles κ(z) from 3-omega frequency-sweep measurements using Conditional Flow Matching / Rectified Flow.

## Motivation

Current thermal inversion methods face a fundamental trade-off:
- **MCMC / Bayesian methods**: Full uncertainty quantification, but extremely slow (hours per sample)
- **MLP / KRR regression**: Fast inference, but point estimates only (no UQ)

**This project** uses Conditional Flow Matching to achieve both: **millisecond inference** with **full posterior distributions** for uncertainty quantification.

## Method

1. **Forward Model**: Differentiable Borca-Tasciuc transfer matrix (PyTorch)
2. **Synthetic Data**: 100K+ (κ(z), V_3ω(f)) pairs with realistic noise
3. **CFM Training**: Learn velocity field v_θ(x_t, t, y) mapping noise → κ posterior
4. **Reflow**: Straighten ODE trajectories for few-step inference
5. **Physics Constraint**: Forward model consistency loss for physical validity
6. **Validation**: Against Burgholzer's thermodynamic resolution limit Δz = 2πz/ln(SNR)

## Project Structure

```
src/thermal_flow/
├── forward/          # Borca-Tasciuc forward model + data generation
├── data/             # PyTorch datasets + transforms
├── baselines/        # Feldman, Tikhonov, KRR, MLP
├── models/           # CFM core: VelocityNet, FlowMatching, Reflow
├── inference/        # ODE sampling + uncertainty quantification
├── evaluation/       # Metrics + resolution limit validation
└── utils/            # Config, logging, W&B integration
```

## Setup

```bash
conda env create -f environment.yml
conda activate thermal-flow
pip install -e .
```

## Usage

```bash
# Generate synthetic dataset
python scripts/generate_dataset.py

# Train baselines
python scripts/train_baseline.py

# Train CFM model
python scripts/train_cfm.py

# Run Reflow iterations
python scripts/reflow.py --checkpoint outputs/cfm/best.pt

# Evaluate all methods
python scripts/evaluate.py --checkpoint outputs/cfm/best.pt
```

## License

MIT
