from otxlearner.registries import _DATASETS, _TRAINERS
from otxlearner.trainers.sinkhorn import SinkhornTrainer
from otxlearner.trainers.dann import DANNTrainer


def test_registries_populated() -> None:
    assert "ihdp" in _DATASETS
    assert "sinkhorn" in _TRAINERS
    assert _TRAINERS["sinkhorn"] is SinkhornTrainer
    assert _TRAINERS["dann"] is DANNTrainer
