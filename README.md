# Thermal CFM

**Conditional Flow Matching as a Universal Framework for Thermal Property Inversion**

A unified Bayesian inversion framework for thermal measurement techniques. By plugging in different forward models, the same CFM pipeline provides fast posterior sampling with uncertainty quantification for any thermal inverse problem.

## Thesis

All thermal measurement techniques share the same mathematical structure:
**y = F[θ] + η** — a well-posed Bayesian inverse problem where CFM posterior sampling is guaranteed to converge (Jeong et al. 2025, Dasgupta et al. 2025).

Current thermal inversion methods face a fundamental gap:
- **MCMC**: full UQ but hours per sample
- **MLP/KRR**: millisecond inference but no UQ

**CFM fills this gap**: millisecond inference with full posterior distributions.

## Contributions

1. **Theory**: All thermal inverse problems satisfy the CFM convergence conditions (C1-C3)
2. **Framework**: Pluggable forward model architecture — same CFM pipeline for Flash / 3ω / TDTR
3. **Experiments**: Systematic validation across three measurement methods
4. **Insight**: CFM posterior width quantitatively matches Burgholzer's resolution limit Δz = 2πz/ln(SNR)

## Supported Methods

| Method | Forward Model | θ | Dim | Phase |
|--------|--------------|---|-----|-------|
| Flash | Parker analytical | α, h | 2 | Phase 1 (MVP) |
| 3-omega | Borca-Tasciuc transfer matrix | κ(z) | ~100 | Phase 2 |
| TDTR | Multilayer frequency-domain | κ, G, cp | 4+ | Phase 2 |

## Architecture

```
┌────────────────────────────────────────────────────┐
│              Thermal CFM Framework                  │
│                                                     │
│  ┌─────────────┐   ┌──────────────────────────┐    │
│  │Forward Model│   │    CFM Pipeline           │    │
│  │  (pluggable)│──→│  VelocityNet + ODE solver │    │
│  │             │   │  (forward-model agnostic) │    │
│  │ • Flash     │   └──────────────────────────┘    │
│  │ • 3-omega   │              │                     │
│  │ • TDTR      │              ▼                     │
│  │ • [yours]   │   ┌──────────────────────────┐    │
│  └─────────────┘   │  Posterior p(θ|y)         │    │
│                     │  • Mean ± uncertainty     │    │
│                     │  • Calibration curves     │    │
│                     │  • Resolution limit check │    │
│                     └──────────────────────────┘    │
└────────────────────────────────────────────────────┘
```

## Setup

```bash
conda env create -f environment.yml
conda activate thermal-flow
pip install -e .
```

## Usage

```bash
# Phase 1: Flash method (start here)
python scripts/generate_dataset.py --config configs/flash.yaml
python scripts/train_cfm.py --config configs/flash.yaml

# Phase 2: 3-omega depth profiling
python scripts/generate_dataset.py --config configs/three_omega.yaml
python scripts/train_cfm.py --config configs/three_omega.yaml

# Evaluate any method
python scripts/evaluate.py --config configs/flash.yaml --checkpoint outputs/flash/best.pt
```

## Adding a New Measurement Method

1. Subclass `ForwardModel` in `src/thermal_flow/forward/`
2. Implement `forward()`, `sample_prior()`, `add_noise()`, and `spec`
3. Register in `forward/__init__.py`
4. Create a config YAML
5. Run the same pipeline — everything else is automatic

## Research Roadmap

```
Phase 1 (3-4 weeks):           Phase 2 (6-8 weeks):
  Flash (2 params, MVP)          3ω depth profiling (~100D)
  + TDTR/FDTR multi-param        + UQ vs Burgholzer limit
  ↓                               ↓
  Validate CFM convergence       Validate at scale

                Phase 3 (semester):
                  IR thermography 3D (~10⁵D)
                  + Sensor arrays
```

## License

MIT
