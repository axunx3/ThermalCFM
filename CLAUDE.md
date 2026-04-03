# Thermal Flow - Project Guide for Claude Code

## Overview
PhD research project: using Conditional Flow Matching (CFM) / Rectified Flow to solve the 3-omega thermal conductivity depth profiling inverse problem.

## Key Terminology
- **3-omega method**: Contact-based thermal measurement using AC-heated metal line, extracts κ(z) from V_3ω(f)
- **Borca-Tasciuc model**: Transfer matrix forward model mapping κ(z) → V_3ω(f), differentiable via PyTorch
- **CFM (Conditional Flow Matching)**: Generative model learning velocity fields v_θ(x_t, t, y) for conditional transport
- **Rectified Flow / Reflow**: Iterative trajectory straightening enabling few-step generation
- **Virtual Wave**: Burgholzer's transform converting diffusion fields to wave fields; conceptual predecessor to Reflow
- **Resolution limit**: Δz = 2π·z / ln(SNR) — fundamental thermodynamic bound on depth resolution

## Project Structure
- `src/thermal_flow/forward/`: Borca-Tasciuc forward model + data generation
- `src/thermal_flow/models/`: CFM, VelocityNet, Reflow, physics loss
- `src/thermal_flow/baselines/`: Feldman, Tikhonov, KRR, MLP
- `src/thermal_flow/inference/`: ODE sampling + uncertainty quantification
- `src/thermal_flow/evaluation/`: Metrics + Burgholzer resolution limit validation
- `configs/`: Hydra/OmegaConf YAML configs
- `scripts/`: Training and evaluation entry points

## Development Commands
```bash
# Environment
conda env create -f environment.yml
conda activate thermal-flow
pip install -e .

# Tests
pytest tests/

# Training
python scripts/generate_dataset.py --config configs/dataset.yaml
python scripts/train_cfm.py --config configs/cfm.yaml
```

## Conventions
- Use PyTorch for all numerical computation (including forward model) to maintain differentiability
- Work in log(κ) space for training (LogKappaTransform)
- Configs managed via OmegaConf/Hydra YAML files
- Experiment tracking via Weights & Biases
- All physical quantities in SI units (m, W, K, Hz)
