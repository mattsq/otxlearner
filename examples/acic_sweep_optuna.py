#!/usr/bin/env python
"""Optuna hyperparameter sweep on ACIC 2016. Runtime <3 min."""
import argparse
import torch
import optuna
from otxlearner.datasets import load_acic
from otxlearner.trainers import SinkhornTrainer


def objective(trial: optuna.Trial) -> float:
    device = torch.device("cpu")
    train, val, _ = load_acic(device=device)
    lam = trial.suggest_float("lambda_max", 1.0, 10.0)
    eps = trial.suggest_float("epsilon", 0.05, 0.5)
    trainer = SinkhornTrainer(lambda_max=lam, epsilon=eps, device=device)
    hist = trainer.fit(train, val, epochs=1)
    return hist["pehe_val"][-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials)
    print("Best", study.best_params)


if __name__ == "__main__":
    main()
