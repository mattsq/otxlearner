"""Reproducible benchmark harness for OTX Learner."""

from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from otxlearner.data import (
    load_criteo_uplift,
    load_ihdp,
    load_twins,
    torchify,
)
from otxlearner.loops import prepare_loaders
from otxlearner.models import MLPEncoder, Sinkhorn
from otxlearner.trainers import SinkhornTrainer
from otxlearner.utils import cross_fit_propensity, policy_risk, pehe, ate

_wandb: Optional[ModuleType]
try:  # pragma: no cover - optional dependency
    _wandb = importlib.import_module("wandb")
except Exception:  # pragma: no cover - no wandb installed
    _wandb = None
wandb: Optional[ModuleType] = _wandb


def _load_dataset(name: str):
    if name == "ihdp":
        return load_ihdp()
    if name == "twins":
        return load_twins()
    if name == "criteo":
        return load_criteo_uplift(nrows=10000)
    raise ValueError(f"unknown dataset {name}")


def _evaluate(
    model: MLPEncoder, loader: DataLoader, epsilon: float, device: torch.device
) -> dict[str, float]:
    model.eval()
    div = Sinkhorn(blur=epsilon).to(device)
    tau_pred: list[torch.Tensor] = []
    tau_true: list[torch.Tensor] = []
    mu0_all: list[torch.Tensor] = []
    mu1_all: list[torch.Tensor] = []
    bal = 0.0
    with torch.no_grad():
        for x, t, _yf, mu0, mu1, _e in loader:
            x, t, mu0, mu1 = x.to(device), t.to(device), mu0.to(device), mu1.to(device)
            feats = model.net(x)
            tau = model.tau_head(feats).squeeze(-1)
            tau_pred.append(tau.cpu())
            tt = mu1 - mu0
            tau_true.append(tt.cpu())
            mu0_all.append(mu0.cpu())
            mu1_all.append(mu1.cpu())
            ft = feats[t.bool()]
            fc = feats[~t.bool()]
            if len(ft) > 0 and len(fc) > 0:
                bal += div(ft, fc).item() * x.size(0)
    tau_p = torch.cat(tau_pred)
    tau_t = torch.cat(tau_true)
    mu0_cat = torch.cat(mu0_all)
    mu1_cat = torch.cat(mu1_all)
    bal /= len(loader.dataset)
    return {
        "pehe": pehe(tau_p, tau_t),
        "ate_error": ate(tau_p, tau_t),
        "policy_risk": policy_risk(tau_p, tau_t, mu0_cat, mu1_cat),
        "balance": bal,
    }


def run_experiment(
    dataset: str, params: dict[str, Any], *, wandb_log: bool = False
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_np = _load_dataset(dataset)
    x_all = np.concatenate([ds_np.train.x, ds_np.val.x, ds_np.test.x])
    t_all = np.concatenate([ds_np.train.t, ds_np.val.t, ds_np.test.t])
    e_all = cross_fit_propensity(x_all, t_all, n_splits=5, seed=0)
    n_tr = len(ds_np.train.x)
    n_val = len(ds_np.val.x)
    e_train = e_all[:n_tr]
    e_val = e_all[n_tr : n_tr + n_val]
    e_test = e_all[n_tr + n_val :]
    ds = torchify(ds_np, (e_train, e_val, e_test))
    train_loader, val_loader = prepare_loaders(ds, batch_size=512, seed=0)
    trainer = SinkhornTrainer(
        ds.train.x.shape[1],
        lr=params.get("lr", 1e-3),
        lambda_max=params.get("lambda_max", 1.0),
        epsilon=params.get("epsilon", 0.05),
        device=device,
    )
    if wandb_log and wandb is not None:
        wandb.init(project="otxlearner-bench", config={"dataset": dataset, **params})
    trainer.fit(train_loader, val_loader, epochs=params.get("epochs", 5))
    test_loader = DataLoader(ds.test, batch_size=512)
    metrics = _evaluate(trainer.model, test_loader, params.get("epsilon", 0.05), device)
    if wandb_log and wandb is not None:
        wandb.log(metrics)
        wandb.finish()
    return {**params, **metrics}


def grid_sweep(
    dataset: str,
    grid: Iterable[dict[str, Any]],
    csv_path: Path,
    *,
    wandb_log: bool = False,
) -> None:
    results: list[dict[str, Any]] = []
    for params in grid:
        results.append(run_experiment(dataset, params, wandb_log=wandb_log))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def optuna_sweep(
    dataset: str, trials: int, csv_path: Path, *, wandb_log: bool = False
) -> None:
    import optuna

    def objective(trial: optuna.Trial) -> float:
        params = {
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "lambda_max": trial.suggest_float("lambda", 1e-2, 10.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 0.01, 0.1),
        }
        result = run_experiment(dataset, params, wandb_log=wandb_log)
        trial.set_user_attr("result", result)
        return result["pehe"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials)
    rows = [t.user_attrs["result"] for t in study.trials]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark sweeps")
    parser.add_argument("dataset", choices=["ihdp", "twins", "criteo"])
    parser.add_argument("sweep", choices=["grid", "optuna"])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--csv", type=Path, default=Path("bench/results.csv"))
    args = parser.parse_args()
    grid = [
        {"lr": 1e-3, "lambda_max": 1.0, "epsilon": 0.05, "epochs": 5},
        {"lr": 5e-4, "lambda_max": 1.0, "epsilon": 0.05, "epochs": 5},
    ]
    if args.sweep == "grid":
        grid_sweep(args.dataset, grid, args.csv, wandb_log=args.wandb)
    else:
        optuna_sweep(args.dataset, args.trials, args.csv, wandb_log=args.wandb)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
