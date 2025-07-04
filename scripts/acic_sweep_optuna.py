#!/usr/bin/env python
"""Optuna hyperparameter sweep on ACIC 2016. Runtime <3 min."""
import argparse
import torch
import optuna
import numpy as np
from otxlearner.data import load_acic, torchify
from otxlearner.trainers import SinkhornTrainer
from otxlearner.utils import cross_fit_propensity
from otxlearner.loops import prepare_loaders


def objective(trial: optuna.Trial) -> float:
    device = torch.device("cpu")
    ds_np = load_acic()
    x_all = np.concatenate([ds_np.train.x, ds_np.val.x, ds_np.test.x])
    t_all = np.concatenate([ds_np.train.t, ds_np.val.t, ds_np.test.t])
    e_all = cross_fit_propensity(x_all, t_all, n_splits=5, seed=0)
    n_tr = ds_np.train.x.shape[0]
    n_val = ds_np.val.x.shape[0]
    e_train = e_all[:n_tr]
    e_val = e_all[n_tr : n_tr + n_val]
    e_test = e_all[n_tr + n_val :]
    ds = torchify(ds_np, (e_train, e_val, e_test))
    for split in (ds.train, ds.val, ds.test):
        split.x = split.x.to(device)
        split.t = split.t.to(device)
        split.yf = split.yf.to(device)
        split.mu0 = split.mu0.to(device)
        split.mu1 = split.mu1.to(device)
        split.e = split.e.to(device)

    train_loader, val_loader = prepare_loaders(ds, batch_size=512, seed=0)
    lam = trial.suggest_float("lambda_max", 1.0, 10.0)
    eps = trial.suggest_float("epsilon", 0.05, 0.5)
    trainer = SinkhornTrainer(
        ds.train.x.shape[1], lambda_max=lam, epsilon=eps, device=device
    )
    hist = trainer.fit(train_loader, val_loader, epochs=1)
    return hist[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials)
    print("Best", study.best_params)


if __name__ == "__main__":
    main()
