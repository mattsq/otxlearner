from __future__ import annotations

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any, Optional
import importlib

import optuna

from .train import train

_wandb: Optional[ModuleType]
try:  # pragma: no cover - optional dependency
    _wandb = importlib.import_module("wandb")
except Exception:
    _wandb = None
wandb: Optional[ModuleType] = _wandb

__all__ = ["run_study", "suggest"]


def suggest(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "lambda_": trial.suggest_float("lambda", 1e-2, 10.0, log=True),
        "blur": trial.suggest_float("blur", 0.01, 0.1),
        "depth": trial.suggest_int("layers", 2, 4),
        "width": trial.suggest_categorical("width", [64, 128, 256]),
    }


def objective(trial: optuna.trial.Trial, root: Path) -> float:
    params = suggest(trial)
    history = train(
        root,
        epochs=1,
        batch_size=32,
        lr=params["lr"],
        lambda_max=params["lambda_"],
        epsilon=params["blur"],
        patience=1,
        depth=params["depth"],
        width=params["width"],
        wandb_log=False,
    )
    return history[-1]


def run_study(
    n_trials: int = 10, root: Path | str = Path.home() / ".cache/otxlearner/ihdp"
) -> optuna.study.Study:
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, Path(root)), n_trials=n_trials)
    if wandb is not None:
        run = wandb.init(project="otxlearner", job_type="optuna")
        run.summary.update({"best_value": study.best_value, **study.best_params})
        run.finish()
    print("Best trial", study.best_trial.number, study.best_value)
    return study


if __name__ == "__main__":  # pragma: no cover - CLI
    run_study()
