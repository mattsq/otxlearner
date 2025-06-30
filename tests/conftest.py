import os
from pathlib import Path
import pytest


@pytest.fixture()
def ihdp_root(tmp_path: Path) -> Path:
    root_env = os.getenv("IHDP_DATA")
    if root_env:
        path = Path(root_env)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return tmp_path
