from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = ["IHDPSplit", "IHDPDataset", "load_ihdp"]

URL_TRAIN = "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"
URL_TEST = "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz"


@dataclass
class IHDPSplit:
    x: npt.NDArray[np.floating]
    t: npt.NDArray[np.floating]
    yf: npt.NDArray[np.floating]
    ycf: npt.NDArray[np.floating]
    mu0: npt.NDArray[np.floating]
    mu1: npt.NDArray[np.floating]


@dataclass
class IHDPDataset:
    train: IHDPSplit
    val: IHDPSplit
    test: IHDPSplit


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    with urllib.request.urlopen(url) as resp:
        dest.write_bytes(resp.read())


def _load_npz(path: Path) -> dict[str, npt.NDArray[np.floating]]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _flatten(features: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    if features.ndim == 3:  # (n, d, r)
        n, d, r = features.shape
        return np.asarray(features.transpose(2, 0, 1).reshape(n * r, d))
    if features.ndim == 2:  # (n, r)
        n, r = features.shape
        return np.asarray(features.T.reshape(n * r))
    raise ValueError(f"Unexpected shape {features.shape}")


def load_ihdp(
    root: str | Path = Path.home() / ".cache" / "otxlearner" / "ihdp",
    *,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> IHDPDataset:
    """Load IHDP with deterministic train/val/test splits."""
    root = Path(root)
    train_path = root / "ihdp_npci_1-100.train.npz"
    test_path = root / "ihdp_npci_1-100.test.npz"
    _download(URL_TRAIN, train_path)
    _download(URL_TEST, test_path)

    train_npz = _load_npz(train_path)
    test_npz = _load_npz(test_path)

    x_train = _flatten(train_npz["x"])
    t_train = _flatten(train_npz["t"])
    yf_train = _flatten(train_npz["yf"])
    ycf_train = _flatten(train_npz["ycf"])
    mu0_train = _flatten(train_npz["mu0"])
    mu1_train = _flatten(train_npz["mu1"])

    x_test = _flatten(test_npz["x"])
    t_test = _flatten(test_npz["t"])
    yf_test = _flatten(test_npz["yf"])
    ycf_test = _flatten(test_npz["ycf"])
    mu0_test = _flatten(test_npz["mu0"])
    mu1_test = _flatten(test_npz["mu1"])

    rng = np.random.default_rng(seed)
    idx = np.arange(x_train.shape[0])
    rng.shuffle(idx)
    val_size = int(len(idx) * val_fraction)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    def split(idx: npt.NDArray[np.int_]) -> IHDPSplit:
        return IHDPSplit(
            x=x_train[idx],
            t=t_train[idx],
            yf=yf_train[idx],
            ycf=ycf_train[idx],
            mu0=mu0_train[idx],
            mu1=mu1_train[idx],
        )

    train = split(train_idx)
    val = split(val_idx)
    test = IHDPSplit(x_test, t_test, yf_test, ycf_test, mu0_test, mu1_test)

    return IHDPDataset(train=train, val=val, test=test)
