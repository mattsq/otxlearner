from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any, Optional
import importlib

import optuna
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .train import train_from_config

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
        "lambda_max": trial.suggest_float("lambda", 1e-2, 10.0, log=True),
        "epsilon": trial.suggest_float("epsilon", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 2, 4),
        "width": trial.suggest_categorical("width", [64, 128, 256]),
    }


def _objective(
    trial: optuna.trial.Trial, cfg_path: Path, data_root: Path | None
) -> float:
    params = suggest(trial)
    overrides = [f"{k}={v}" for k, v in params.items()]
    if data_root is not None:
        overrides.append(f"data_root={str(data_root.expanduser())}")
    trial_dir = Path("runs") / f"trial_{trial.number}"
    overrides.append(f"log_dir={trial_dir}")
    with initialize_config_dir(
        config_dir=str(cfg_path.parent.resolve()), version_base=None
    ):
        cfg = compose(config_name=cfg_path.stem, overrides=overrides)
    history = train_from_config(cfg)  # type: ignore[arg-type]
    return history[-1]


def run_study(
    n_trials: int = 10,
    cfg_path: Path | str = Path("configs/ihdp.yaml"),
    *,
    data_root: Path | None = None,
    wandb_log: bool = False,
) -> optuna.study.Study:
    cfg = Path(cfg_path)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: _objective(t, cfg, data_root), n_trials=n_trials)
    if wandb_log and wandb is not None:
        run = wandb.init(project="otxlearner", job_type="optuna")
        run.summary.update({"best_value": study.best_value, **study.best_params})
        best_overrides = [f"{k}={v}" for k, v in study.best_params.items()]
        if data_root is not None:
            best_overrides.append(f"data_root={str(data_root.expanduser())}")
        with initialize_config_dir(
            config_dir=str(cfg.parent.resolve()), version_base=None
        ):
            best_cfg = compose(config_name=cfg.stem, overrides=best_overrides)
        cfg_file = Path("runs") / "best_config.yaml"
        OmegaConf.save(best_cfg, cfg_file)
        artifact = wandb.Artifact("best_config", type="config")
        artifact.add_file(str(cfg_file))
        run.log_artifact(artifact)
        run.finish()
    return study


if __name__ == "__main__":  # pragma: no cover - CLI
    run_study()
