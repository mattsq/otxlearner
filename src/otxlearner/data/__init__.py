from .ihdp import IHDPDataset, IHDPSplit, load_ihdp
from .twins import TwinsDataset, TwinsSplit, load_twins
from .acic import ACICDataset, ACICSplit, load_acic
from .criteo import CriteoDataset, CriteoSplit, load_criteo_uplift
from .tabular import TabularDataset, TabularSplit, load_tabular
from .torch_adapter import TorchIHDP, TorchSplit, torchify

__all__ = [
    "IHDPDataset",
    "IHDPSplit",
    "load_ihdp",
    "ACICDataset",
    "ACICSplit",
    "load_acic",
    "TwinsDataset",
    "TwinsSplit",
    "load_twins",
    "CriteoDataset",
    "CriteoSplit",
    "load_criteo_uplift",
    "TabularDataset",
    "TabularSplit",
    "load_tabular",
    "TorchIHDP",
    "TorchSplit",
    "torchify",
]

from pathlib import Path
from ..registries import register_dataset


@register_dataset("ihdp")
def get_ihdp(root: str | Path = Path.home() / ".cache/otxlearner/ihdp") -> IHDPDataset:
    return load_ihdp(root)


@register_dataset("twins")
def get_twins(
    root: str | Path = Path.home() / ".cache/otxlearner/twins",
) -> TwinsDataset:
    return load_twins(root)


@register_dataset("acic")
def get_acic(root: str | Path = Path.home() / ".cache/otxlearner/acic") -> ACICDataset:
    return load_acic(root)


@register_dataset("criteo")
def get_criteo(
    root: str | Path = Path.home() / ".cache/otxlearner/criteo",
) -> CriteoDataset:
    return load_criteo_uplift(root)
