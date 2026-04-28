"""Emotion2vec speech-emotion-representation teacher.

Backed by `emotion2vec`_ via the FunASR_ ``AutoModel`` wrapper -- the
canonical loader maintained by the model authors. We call ``generate``
once per chunked batch with ``granularity='utterance'`` and
``extract_embedding=True``, which returns a 768-dim vector per chunk
(for the ``iic/emotion2vec_base`` checkpoint).

Output shape: ``(B, N_chunks, target_dim)``.

Default model is ``iic/emotion2vec_base`` -- the universal pretrained
emotion representation (no fine-tuning), 768-d. The ``_plus_*`` variants
are 9-class classifiers and only useful with ``extract_embedding=True``
to recover the underlying features; for distillation the base
representation is the right target.

.. _emotion2vec: https://github.com/ddlBoJack/emotion2vec
.. _FunASR: https://github.com/modelscope/FunASR
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class Emotion2vecConfig:
    model_id: str = "emotion2vec/emotion2vec_base"
    target_dim: int = 768
    pooled: str = "chunked"
    sample_rate: int = 16000
    target_layernorm: bool = True
    # FunASR's AutoModel hub: "ms" (modelscope, China) or "hf" (HuggingFace).
    # HF works without a CN account.
    hub: str = "hf"


def _coerce_config(cfg: Any) -> Emotion2vecConfig:
    if isinstance(cfg, Emotion2vecConfig):
        return cfg
    fields = (
        "model_id",
        "target_dim",
        "pooled",
        "sample_rate",
        "target_layernorm",
        "hub",
    )
    if hasattr(cfg, "kind"):
        return Emotion2vecConfig(
            **{f: getattr(cfg, f) for f in fields if hasattr(cfg, f) and getattr(cfg, f) is not None}
        )
    return Emotion2vecConfig(**{f: cfg[f] for f in fields if f in cfg})


def _load_emotion2vec(model_id: str, hub: str, device: torch.device | str):
    """Load the FunASR ``AutoModel`` wrapper around emotion2vec."""
    from funasr import AutoModel  # type: ignore

    # Force registration of the ``Emotion2vec`` model class. FunASR's
    # ``AutoModel`` looks up ``tables.model_classes["Emotion2vec"]``, which
    # is only populated when ``funasr.models.emotion2vec.model`` is
    # imported (the module applies a ``@tables.register`` decorator at
    # import time). Worker processes spawned by ``torch_xla.xmp.spawn``
    # don't trigger FunASR's lazy auto-discovery, so without this import
    # we get ``AssertionError: iic/emotion2vec_base is not registered``.
    import funasr.models.emotion2vec.model  # noqa: F401

    return AutoModel(
        model=model_id,
        hub=hub,
        device=str(device) if not isinstance(device, str) else device,
        disable_update=True,
        disable_pbar=True,
        disable_log=True,
    )


class Emotion2vecTeacher(nn.Module):
    """Per-chunk emotion2vec teacher.

    The wrapped FunASR model handles its own preprocessing (16 kHz raw
    waveform in, internal LayerNorm + AudioEncoder + transformer), so we
    just hand it a flat list of per-chunk numpy arrays and reshape the
    returned utterance embeddings back to ``(B, N, D)``.
    """

    def __init__(
        self,
        cfg: Any,
        *,
        device: torch.device | str = "cpu",
        model: Any | None = None,
    ) -> None:
        super().__init__()
        self.cfg = _coerce_config(cfg)
        self.target_dim = int(self.cfg.target_dim)
        self.target_layernorm = bool(self.cfg.target_layernorm)
        self._device = torch.device(device) if not isinstance(device, torch.device) else device
        if model is None:
            model = _load_emotion2vec(self.cfg.model_id, self.cfg.hub, self._device)
        # Stash as a plain attribute (not nn.Module submodule) -- FunASR's
        # AutoModel wraps a torch.nn.Module but the wrapper itself isn't
        # one, and we don't want torch to chase its parameters.
        object.__setattr__(self, "model", model)

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        *,
        chunk_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if audio.dim() != 3:
            raise ValueError(f"expected (B, N_chunks, T), got {tuple(audio.shape)}")
        b, n, t = audio.shape
        if chunk_mask is None:
            chunk_mask = torch.ones((b, n), dtype=torch.bool, device=audio.device)

        # FunASR's emotion2vec.inference loops over the input list one
        # element at a time (internal batch size = 1), so flattening
        # (B, N) is correct -- we lose nothing by batching at the call
        # site. Convert to numpy float32 on CPU; the underlying model
        # handles the host-to-device copy itself.
        flat_np = (
            audio.reshape(b * n, t).to(torch.float32).cpu().numpy()
        )
        wav_list: list[np.ndarray] = [flat_np[i] for i in range(flat_np.shape[0])]

        results = self.model.generate(
            wav_list,
            granularity="utterance",
            extract_embedding=True,
            disable_pbar=True,
        )
        if len(results) != b * n:
            raise RuntimeError(
                f"emotion2vec returned {len(results)} embeddings for {b * n} chunks"
            )

        embs = np.stack([r["feats"] for r in results], axis=0)  # (B*N, D)
        if embs.ndim != 2 or embs.shape[-1] != self.target_dim:
            raise ValueError(
                f"emotion2vec returned shape {embs.shape}; "
                f"expected (B*N, {self.target_dim})"
            )

        out = torch.from_numpy(embs).to(audio.device, dtype=torch.float32).reshape(b, n, self.target_dim)
        if self.target_layernorm:
            out = torch.nn.functional.layer_norm(out, (self.target_dim,))
        out = out * chunk_mask.unsqueeze(-1).to(out.dtype)
        return out

    def __call__(  # type: ignore[override]
        self,
        audio: torch.Tensor,
        *,
        chunk_mask: torch.Tensor | None = None,
        **ctx: Any,
    ) -> torch.Tensor:
        return self.forward(audio, chunk_mask=chunk_mask, **ctx)


__all__ = ["Emotion2vecConfig", "Emotion2vecTeacher"]
