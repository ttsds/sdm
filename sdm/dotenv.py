"""Tiny `.env` loader so `WANDB_API_KEY` and friends propagate without a
dotenv dependency on TPU hosts.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | os.PathLike = ".env", *, override: bool = False) -> None:
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
