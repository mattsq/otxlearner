# OTX Learner — Agent Guidelines

## Overview
This repository explores a "Sinkhorn-penalised X-Net" for causal inference. The long-form description lives in `Prompt.md`. Follow these guidelines when extending the codebase.

## High-level objectives
- Implement a minimal baseline using an MLP encoder and Sinkhorn penalty.
- Achieve decreasing validation PEHE on the IHDP dataset within about an hour.
- Only after the baseline works should you add more advanced features.

## Suggested repository layout
```
src/otxlearner/
    data/
    models/
    train.py
    evaluate.py
    utils/
configs/
notebooks/
tests/
requirements.txt
Makefile
.github/workflows/
```
Maintain future contributions in this structure.

## Development practices
- Format and lint with **ruff**, **black**, and **mypy --strict**. Always run
  `ruff check src tests`, `black --check src tests`, `mypy --strict src` and
  `pytest -vv` locally before committing so the checks match CI.
- Install the package in editable mode via `pip install -e .` before running tests.
- Add unit tests for each component, including a one‑epoch IHDP smoke test.
- Provide GitHub Actions CI for Python 3.10 and 3.11 on CPU and CUDA 11.8.
- Log λ, ε, PEHE and ATE via MLflow or Weights & Biases.
- Respect the MIT licence.

## Training tips
- Encoder architecture: `[input → 256 → 128 → 64]` with LayerNorm and GELU.
- Clip propensities so `ε_prop ≤ e ≤ 1 - ε_prop`.
- Ramp λ with a cosine schedule during the first 10 % of epochs.
- Use blur ≈ 0.05 on L2‑normalised features.
- Aim for batch size around 512; use gradient accumulation otherwise.
- Early stopping: `val_PEHE_proxy = MSE(D̂, τ̂) + 0.1·L_bal`.

## Possible extensions
1. Continuous normalising‑flow encoder.
2. Gradient‑reversal (DANN) variant.
3. Unbalanced OT via `ott.solvers.linear.sinkhorn`.
4. Contrastive τ pre‑text task.
5. Diffusion‑based counterfactual augmentation.
6. Policy value head for decision loss.

## Scratchpad
Create a `scratchpad.md` file in the repository root. Use it as **write-only** storage for useful findings or ideas. Do not read back previous notes when choosing actions, but keep appending anything that may help future agents.
