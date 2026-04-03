# Thermal CFM - Project Guide

## Overview
PhD research: Conditional Flow Matching as a **universal framework** for thermal property inversion. Not "CFM for 3ω" — it's "CFM as a paradigm shift for all thermal inverse problems."

The key insight: all thermal measurements share y = F[θ] + η, satisfying CFM convergence conditions (C1: F continuous, C2: bounded prior, C3: positive-density noise).

## Key Terminology
- **Forward model F**: maps thermal parameters θ to measurements y (pluggable)
- **CFM**: learns velocity field v_θ(x_t, t, y) for conditional posterior sampling
- **Reflow**: trajectory straightening for few-step inference (data-driven Virtual Wave)
- **Burgholzer resolution limit**: Δz = 2πz/ln(SNR), the physical validation target
- **UQ**: uncertainty quantification — the core advantage over MLP/KRR

## Architecture: Pluggable Forward Models
- `forward/base.py`: `ForwardModel` ABC — implement `forward()`, `sample_prior()`, `add_noise()`, `spec`
- `forward/flash.py`: Flash method (Phase 1, 2 params)
- `forward/borca_tasciuc.py`: 3-omega (Phase 2, ~100 dims)
- `forward/tdtr.py`: TDTR (Phase 2, 4+ params)
- Everything downstream (CFM, inference, evaluation) is forward-model agnostic

## Research Phases
- Phase 1: Flash → validate CFM convergence (3-4 weeks)
- Phase 2: 3ω + TDTR → validate UQ vs physical limits (6-8 weeks)
- Phase 3: IR thermography 3D → validate large-scale (semester)

## Config-Driven Workflow
```bash
# Same pipeline, different config = different method
python scripts/generate_dataset.py --config configs/flash.yaml
python scripts/train_cfm.py --config configs/flash.yaml
python scripts/evaluate.py --config configs/flash.yaml --checkpoint ...
```

## Development Commands
```bash
conda env create -f environment.yml && conda activate thermal-flow
pip install -e .
pytest tests/
```

## Conventions
- PyTorch throughout (differentiability for physics loss)
- Log-space for parameters spanning orders of magnitude
- OmegaConf YAML configs, W&B experiment tracking
- SI units (m, W, K, Hz)
