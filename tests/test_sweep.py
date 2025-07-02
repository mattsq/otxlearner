from pathlib import Path

from otxlearner.sweep import run_study


def test_sweep_runs_one_trial(ihdp_root: Path) -> None:
    study = run_study(n_trials=1, cfg_path="configs/ihdp.yaml", data_root=ihdp_root)
    assert len(study.trials) == 1
