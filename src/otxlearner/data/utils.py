from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
import tarfile
import zipfile

__all__ = ["download", "download_and_extract", "sha256sum"]


def sha256sum(path: Path) -> str:
    hash_obj = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def download(url: str, dest: Path, *, sha256: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if sha256 is None or sha256sum(dest) == sha256:
            return
        dest.unlink()
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("Checksum mismatch")
    dest.write_bytes(data)


def download_and_extract(url: str, dest: Path, *, sha256: str | None = None) -> Path:
    download(url, dest, sha256=sha256)
    out_dir = dest.parent
    if dest.suffix == ".zip":
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(out_dir)
    elif dest.suffix in {".gz", ".tgz"} or dest.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(dest) as tf:
            tf.extractall(out_dir)
    return out_dir
