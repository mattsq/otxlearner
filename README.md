# OTX Learner

This project implements a Sinkhorn-penalised X-Net architecture for causal inference.
Refer to `Prompt.md` for the full research notes.

This repository currently contains a minimal project skeleton. Code lives under
`src/otxlearner`, configurations under `configs/`, and tests under `tests/`.

## Dataset loaders

`load_ihdp()` downloads the public IHDP benchmark and returns deterministic
train/val/test splits. By default data are cached under `~/.cache/otxlearner/ihdp`.

`load_twins()` downloads the Twins benchmark if needed and returns deterministic
train/val/test splits. By default data are cached under
`~/.cache/otxlearner/twins`.

`load_acic()` supports the ACIC 2016 and 2018 benchmarks. The dataset is
downloaded on first use and cached under `~/.cache/otxlearner/acic`.

## Quick start

Install the required packages and run a short training session on IHDP. The
dataset is downloaded automatically on first use.

```bash
python -m pip install torch geomloss  # plus other deps in requirements.txt
python -m pip install -r requirements.txt
python -m pip install -e .

# train for a few epochs and log metrics under `runs/ihdp`
python -m otxlearner.train --epochs 5 --log-dir runs/ihdp

# evaluate a saved checkpoint (if you saved `model.pt` during training)
python -m otxlearner.evaluate model.pt --data-root ~/.cache/otxlearner/ihdp \
    --csv results.csv --plot results.png
```

The [training_curves.ipynb](notebooks/training_curves.ipynb) notebook shows how
to visualise the TensorBoard logs produced during training.

## Experiment tracking

Set the `WANDB_API_KEY` environment variable or run `wandb login` to enable
Weights & Biases logging via `--wandb` on the train and evaluate scripts.
