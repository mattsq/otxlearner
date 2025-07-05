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

### Train on a CSV file

After loading the data you can train a `SinkhornTrainer` using the generic
loader:

```python
import numpy as np
from otxlearner.data import load_tabular, torchify
from otxlearner.trainers import SinkhornTrainer
from otxlearner.utils import cross_fit_propensity
from otxlearner.loops import prepare_loaders

ds_np = load_tabular(
    "data.csv",
    features=["f0", "f1", "f2"],
    treatment="t",
    outcome="y",
)

x_all = np.concatenate([ds_np.train.x, ds_np.val.x, ds_np.test.x])
t_all = np.concatenate([ds_np.train.t, ds_np.val.t, ds_np.test.t])
e_all = cross_fit_propensity(x_all, t_all, seed=0)
n_tr = len(ds_np.train.x)
n_val = len(ds_np.val.x)
ds = torchify(ds_np, (e_all[:n_tr], e_all[n_tr : n_tr + n_val], e_all[n_tr + n_val :]))

train_loader, val_loader = prepare_loaders(ds, batch_size=512)
trainer = SinkhornTrainer(ds.train.x.shape[1])
history = trainer.fit(train_loader, val_loader, epochs=5)
```

## Experiment tracking

Set the `WANDB_API_KEY` environment variable or run `wandb login` to enable Weights & Biases logging via `--wandb` on the train and evaluate scripts.

