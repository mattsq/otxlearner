"""Command-line interface for OTX Learner."""

from __future__ import annotations

import argparse
from pathlib import Path
import inspect
from typing import Any, Callable, cast
import numpy as np

# ensure dataset and trainer registries are populated
from . import data  # noqa: F401
from .trainers import dann, sinkhorn  # noqa: F401

from .registries import _DATASETS, _TRAINERS
from .trainers.base import BaseTrainer
from .data.types import DatasetProtocol
from .data.torch_adapter import torchify
from .utils import cross_fit_propensity
from .loops import prepare_loaders

__all__ = ["main", "train_from_config"]


def _trainer_kwargs(args: argparse.Namespace, cls: type[Any]) -> dict[str, object]:
    params = inspect.signature(cast(Callable[..., object], cls.__init__)).parameters
    return {
        name: getattr(args, name)
        for name in params
        if name != "self" and hasattr(args, name)
    }


def main(argv: list[str] | None = None) -> list[float]:
    """Entry point for ``python -m otxlearner.train``."""

    parser = argparse.ArgumentParser(description="Train OTX Learner")
    parser.add_argument("dataset", choices=sorted(_DATASETS))
    parser.add_argument("trainer", choices=sorted(_TRAINERS))
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_max", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--unbalanced", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    root = args.data_root or Path.home() / f".cache/otxlearner/{args.dataset}"
    ds_np = cast(DatasetProtocol, _DATASETS[args.dataset](root=root))
    x_all = np.concatenate([ds_np.train.x, ds_np.val.x, ds_np.test.x])
    t_all = np.concatenate([ds_np.train.t, ds_np.val.t, ds_np.test.t])
    e_all = cross_fit_propensity(x_all, t_all, n_splits=5, seed=args.seed)
    n_tr = ds_np.train.x.shape[0]
    n_val = ds_np.val.x.shape[0]
    e_train = e_all[:n_tr]
    e_val = e_all[n_tr : n_tr + n_val]
    e_test = e_all[n_tr + n_val :]
    ds = torchify(ds_np, (e_train, e_val, e_test))

    train_loader, val_loader = prepare_loaders(ds, args.batch_size, seed=args.seed)

    trainer_cls = cast(type[Any], _TRAINERS[args.trainer])
    kwargs = _trainer_kwargs(args, trainer_cls)
    kwargs["input_dim"] = ds.train.x.shape[1]
    trainer = cast(BaseTrainer, trainer_cls(**kwargs))
    history = trainer.fit(train_loader, val_loader, epochs=args.epochs)
    return history


def train_from_config(cfg: Any) -> list[float]:
    """Run training from a configuration object."""

    trainer = "sinkhorn"
    if getattr(cfg, "dann", False):
        trainer = "dann"
    argv = [
        str(cfg.data),
        trainer,
        "--data_root",
        str(cfg.data_root),
        "--epochs",
        str(cfg.epochs),
        "--batch_size",
        str(cfg.batch_size),
        "--lr",
        str(cfg.lr),
        "--lambda_max",
        str(cfg.lambda_max),
        "--epsilon",
        str(cfg.epsilon),
        "--device",
        str(cfg.device),
    ]
    if getattr(cfg, "unbalanced", False):
        argv.append("--unbalanced")
    return main(argv)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
