# OTX Learner

This project implements a Sinkhorn-penalised X-Net architecture for causal inference.
Refer to `Prompt.md` for the full research notes.

This repository currently contains a minimal project skeleton. Code lives under
`src/`, configurations under `configs/`, and tests under `tests/`.

## Dataset loaders

`load_ihdp()` downloads the public IHDP benchmark and returns deterministic
train/val/test splits. By default data are cached under `~/.cache/otxlearner/ihdp`.

`load_twins()` expects a `twins.npz` archive with arrays `x`, `t`, `y0`, and
`y1`. It returns deterministic train/val/test splits and caches the dataset under
`~/.cache/otxlearner/twins`.
