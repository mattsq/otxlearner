from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .utils import download

__all__ = ["TwinsSplit", "TwinsDataset", "load_twins"]

URL = "https://example.com/twins.npz"
SHA = "0000000000000000000000000000000000000000000000000000000000000000"


@dataclass
class TwinsSplit:
    x: npt.NDArray[np.float64]
    t: npt.NDArray[np.float64]
    yf: npt.NDArray[np.float64]
    ycf: npt.NDArray[np.float64]
    mu0: npt.NDArray[np.float64]
    mu1: npt.NDArray[np.float64]


@dataclass
class TwinsDataset:
    train: TwinsSplit
    val: TwinsSplit
    test: TwinsSplit


def _download(url: str, dest: Path, sha: str) -> None:
    download(url, dest, sha256=sha)


def _load_npz(path: Path) -> dict[str, npt.NDArray[np.float64]]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def load_twins(
    root: str | Path = Path.home() / ".cache" / "otxlearner" / "twins",
    *,
    validate: bool | None = None,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> TwinsDataset:
    """Load Twins with deterministic train/val/test splits."""
    root = Path(root)
    path = root / "twins.npz"
    if validate is None:
        validate = root == Path.home() / ".cache" / "otxlearner" / "twins"
    if not path.exists() or validate:
        _download(URL, path, SHA)

    data = _load_npz(path)
    x = np.asarray(data["x"])
    t = np.asarray(data["t"])
    y0 = np.asarray(data["y0"])
    y1 = np.asarray(data["y1"])

    rng = np.random.default_rng(seed)
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)

    val_size = int(len(idx) * val_fraction)
    train_idx = idx[: -val_size * 2]
    val_idx = idx[-val_size * 2 : -val_size]
    test_idx = idx[-val_size:]

    def split(i: npt.NDArray[np.int_]) -> TwinsSplit:
        return TwinsSplit(
            x=x[i],
            t=t[i],
            yf=np.where(t[i] == 1, y1[i], y0[i]),
            ycf=np.where(t[i] == 1, y0[i], y1[i]),
            mu0=y0[i],
            mu1=y1[i],
        )

    train = split(train_idx)
    val = split(val_idx)
    test = split(test_idx)

    return TwinsDataset(train=train, val=val, test=test)
