# OTX Learner [![PyPI](https://img.shields.io/pypi/v/otxlearner.svg)](https://pypi.org/project/otxlearner/)

OTX Learner is an experimental implementation of a **Sinkhorn‑penalised X‑Net** for counterfactual inference with PyTorch. It balances treated and control representations using an entropic optimal‑transport divergence rather than an adversarial critic.

The code lives under `src/otxlearner` and includes dataset loaders, models and training scripts.

```python
from otxlearner.cli import main

# run a short IHDP experiment
main(["ihdp", "sinkhorn", "--epochs", "5"])
```

See the [documentation](https://otxlearner.readthedocs.io/) for a quick-start guide, API reference and research notes.

## Dataset loaders

`load_ihdp()` downloads the public IHDP benchmark and returns deterministic train/val/test splits. By default data are cached under `~/.cache/otxlearner/ihdp`.

`load_twins()` downloads the Twins benchmark if needed and returns deterministic train/val/test splits. By default data are cached under `~/.cache/otxlearner/twins`.

`load_acic()` supports the ACIC 2016 and 2018 benchmarks. The dataset is downloaded on first use and cached under `~/.cache/otxlearner/acic`.

## Quick start

Install the required packages and run a short training session on IHDP.

```bash
python -m pip install torch geomloss
python -m pip install -r requirements.txt
python -m pip install -e .

python -m otxlearner.train ihdp sinkhorn --epochs 5 --log-dir runs/ihdp
```

The [training_curves.ipynb](notebooks/training_curves.ipynb) notebook shows how to visualise the TensorBoard logs produced during training.

## Experiment tracking

Set the `WANDB_API_KEY` environment variable or run `wandb login` to enable Weights & Biases logging via `--wandb` on the train and evaluate scripts.

