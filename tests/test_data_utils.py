from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from otxlearner.data.utils import download, download_and_extract, sha256sum


def test_sha256sum(tmp_path: Path) -> None:
    p = tmp_path / "file.txt"
    data = b"hello"
    p.write_bytes(data)
    assert sha256sum(p) == hashlib.sha256(data).hexdigest()


def test_download_file_uri(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    src.write_text("data")
    dest = tmp_path / "dest.txt"
    download(src.as_uri(), dest)
    assert dest.read_text() == "data"


def test_download_checksum_mismatch(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    src.write_text("data")
    dest = tmp_path / "dest.txt"
    with pytest.raises(ValueError):
        download(src.as_uri(), dest, sha256="bad")


def test_download_and_extract_zip(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "a.txt").write_text("x")
    archive = tmp_path / "src.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(src_dir / "a.txt", "a.txt")
    out = download_and_extract(archive.as_uri(), tmp_path / "dl.zip")
    assert (out / "a.txt").read_text() == "x"
