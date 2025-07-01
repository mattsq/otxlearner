from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.sweep import run_study


def test_sweep_runs_one_trial(ihdp_root: Path) -> None:
    study = run_study(n_trials=1, root=ihdp_root)
    assert len(study.trials) == 1
