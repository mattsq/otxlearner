from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .utils import download

__all__ = ["ACICSplit", "ACICDataset", "load_acic"]

URL_2016 = "https://example.com/acic_2016.npz"
URL_2018 = "https://example.com/acic_2018.npz"
SHA_2016 = "0000000000000000000000000000000000000000000000000000000000000000"
SHA_2018 = "0000000000000000000000000000000000000000000000000000000000000000"


@dataclass
class ACICSplit:
    x: npt.NDArray[np.float64]
    t: npt.NDArray[np.float64]
    yf: npt.NDArray[np.float64]
    ycf: npt.NDArray[np.float64]
    mu0: npt.NDArray[np.float64]
    mu1: npt.NDArray[np.float64]


@dataclass
class ACICDataset:
    train: ACICSplit
    val: ACICSplit
    test: ACICSplit


def _download(url: str, dest: Path, sha: str) -> None:
    download(url, dest, sha256=sha)


def _load_npz(path: Path) -> dict[str, npt.NDArray[np.float64]]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def load_acic(
    root: str | Path = Path.home() / ".cache" / "otxlearner" / "acic",
    *,
    year: int = 2016,
    validate: bool | None = None,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> ACICDataset:
    """Load ACIC with deterministic train/val/test splits."""
    root = Path(root)
    if year == 2016:
        url = URL_2016
        sha = SHA_2016
        fname = "acic_2016.npz"
    else:
        url = URL_2018
        sha = SHA_2018
        fname = "acic_2018.npz"
    path = root / fname
    legacy = root / "acic.npz"
    if legacy.exists() and not path.exists():
        path = legacy
    if validate is None:
        validate = root == Path.home() / ".cache" / "otxlearner" / "acic"
    if not path.exists() or validate:
        _download(url, path, sha)

    data = _load_npz(path)
    x = np.asarray(data["x"])
    t = np.asarray(data["t"])
    y0 = np.asarray(data["y0"])
    y1 = np.asarray(data["y1"])

    rng = np.random.default_rng(seed)
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)

    val_size = int(len(idx) * val_fraction)
    test_size = int(len(idx) * test_fraction)
    train_idx = idx[: -val_size - test_size]
    val_idx = idx[-val_size - test_size : -test_size]
    test_idx = idx[-test_size:]

    def split(i: npt.NDArray[np.int_]) -> ACICSplit:
        return ACICSplit(
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

    return ACICDataset(train=train, val=val, test=test)
