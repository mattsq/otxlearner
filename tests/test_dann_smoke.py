from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.train import train


def test_train_dann_smoke(ihdp_root: Path) -> None:
    history = train(
        root=ihdp_root,
        epochs=2,
        batch_size=32,
        lr=1e-3,
        lambda_max=0.1,
        epsilon=0.05,
        patience=1,
        log_dir=ihdp_root / "logs",
        dann=True,
    )
    assert len(history) >= 2
    assert history[-1] <= history[0]
