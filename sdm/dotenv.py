"""Tiny `.env` loader so `WANDB_API_KEY` and friends propagate without a
dotenv dependency on TPU hosts.
"""

from __future__ import annotations

import os
from pathlib import Path


_HF_TOKEN_ENV_KEYS = ("HF_TOKEN", "hf_token", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")


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


def hf_token() -> str | None:
    for key in _HF_TOKEN_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            return value
    return None


def hf_token_kwargs() -> dict[str, str]:
    token = hf_token()
    return {"token": token} if token else {}
