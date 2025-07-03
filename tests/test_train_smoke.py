from __future__ import annotations

from pathlib import Path

from otxlearner.cli import main as cli_main


def test_train_smoke(ihdp_root: Path) -> None:
    history = cli_main(
        [
            "ihdp",
            "sinkhorn",
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
            "--epsilon",
            "0.05",
        ]
    )
    assert len(history) >= 2
