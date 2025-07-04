from __future__ import annotations

from pathlib import Path

from otxlearner.cli import main as cli_main


def test_train_dann_smoke(ihdp_root: Path, fast_smoke: None) -> None:
    history = cli_main(
        [
            "ihdp",
            "dann",
            "--data_root",
            str(ihdp_root),
            "--epochs",
            "2",
            "--batch_size",
            "32",
            "--lr",
            "1e-3",
            "--lambda_max",
            "0.1",
        ]
    )
    assert len(history) >= 2
