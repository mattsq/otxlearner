# OTX Learner [![PyPI](https://img.shields.io/pypi/v/otxlearner.svg)](https://pypi.org/project/otxlearner/)


OTX Learner is a minimal **Sinkhorn‑penalised X‑Net** for counterfactual inference in PyTorch. Using optimal transport instead of an adversarial critic leads to faster, stabler convergence on small benchmarks.

The code lives under `src/otxlearner` and includes dataset loaders, models and training scripts.

```python
from otxlearner.cli import main

# run a short IHDP experiment
main(["ihdp", "sinkhorn", "--epochs", "5"])
```

See the [documentation](https://otxlearner.readthedocs.io/) for a quick-start guide, API reference and research notes.

## Quick start

Install the required packages and run a short training session on IHDP.

```bash
python -m pip install -e .[bench]

python -m otxlearner.train ihdp sinkhorn --epochs 5 --log-dir runs/ihdp
```

The [training_curves.ipynb](examples/training_curves.ipynb) notebook shows how to visualise the TensorBoard logs produced during training.

## Datasets

`load_ihdp()` downloads the public IHDP benchmark and returns deterministic train/val/test splits. By default data are cached under `~/.cache/otxlearner/ihdp`.

`load_twins()` downloads the Twins benchmark if needed and returns deterministic train/val/test splits. By default data are cached under `~/.cache/otxlearner/twins`.

`load_acic()` supports the ACIC 2016 and 2018 benchmarks. The dataset is downloaded on first use and cached under `~/.cache/otxlearner/acic`.

`load_tabular()` creates train/val/test splits from a CSV file or `pandas.DataFrame`. Specify the feature, treatment and outcome column names:

```python
from otxlearner.data import load_tabular

ds = load_tabular(
    "data.csv",
    features=["f0", "f1", "f2"],
    treatment="t",
    outcome="y",
    val_fraction=0.2,
    test_fraction=0.1,
)
```

## Experiment tracking

Set the `WANDB_API_KEY` environment variable or run `wandb login` to enable Weights & Biases logging via `--wandb` on the train and evaluate scripts.

