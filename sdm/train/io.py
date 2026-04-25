"""Checkpoint I/O that works for both local paths and `gs://` URIs.

GCS support shells out to `gsutil`, which is preinstalled on every TPU VM and
on local dev hosts that have run `gcloud init`. This avoids pulling in
`google-cloud-storage` / `gcsfs` as a hard dependency.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch


def is_gcs(path: str | os.PathLike) -> bool:
    return str(path).startswith("gs://")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def save_state(state: dict[str, Any], path: str) -> None:
    """Save a torch state dict. `path` may be local or `gs://...`."""
    if is_gcs(path):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp = f.name
        try:
            torch.save(state, tmp)
            _run(["gsutil", "-q", "cp", tmp, path])
        finally:
            Path(tmp).unlink(missing_ok=True)
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_state(path: str, map_location: str = "cpu") -> dict[str, Any]:
    """Load a torch state dict. `path` may be local or `gs://...`."""
    if is_gcs(path):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp = f.name
        try:
            _run(["gsutil", "-q", "cp", path, tmp])
            return torch.load(tmp, map_location=map_location)
        finally:
            Path(tmp).unlink(missing_ok=True)
    return torch.load(path, map_location=map_location)


def exists(path: str) -> bool:
    if is_gcs(path):
        try:
            _run(["gsutil", "-q", "stat", path])
            return True
        except subprocess.CalledProcessError:
            return False
    return Path(path).exists()


_STEP_RE = re.compile(r"step-(\d+)\.pt$")


def latest_checkpoint(ckpt_dir: str) -> str | None:
    """Find `ckpt_dir/latest.pt` if present, else the highest `step-NNNNN.pt`."""
    latest = f"{ckpt_dir.rstrip('/')}/latest.pt"
    if exists(latest):
        return latest
    if is_gcs(ckpt_dir):
        try:
            out = _run(["gsutil", "ls", f"{ckpt_dir.rstrip('/')}/step-*.pt"]).stdout
        except subprocess.CalledProcessError:
            return None
        candidates = [line.strip() for line in out.splitlines() if line.strip()]
    else:
        d = Path(ckpt_dir)
        if not d.is_dir():
            return None
        candidates = [str(p) for p in d.glob("step-*.pt")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(_STEP_RE.search(p).group(1)))


def copy(src: str, dst: str) -> None:
    """Copy file across local <-> gs:// boundary."""
    if is_gcs(src) or is_gcs(dst):
        _run(["gsutil", "-q", "cp", src, dst])
        return
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
