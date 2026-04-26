"""mHuBERT backbone wrapper used by the post-pivot distillation pipeline.

The distillation configs operate on raw audio chunk tensors rather than
NeuCodec token ids. This module provides a thin wrapper around a HuggingFace
audio backbone so both training and probing can share the same chunk-level
latent extraction path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel

from sdm.dotenv import hf_token_kwargs


DEFAULT_BACKBONE_MODEL_ID = "utter-project/mHuBERT-147"


@dataclass
class BackboneConfig:
    model_id: str = DEFAULT_BACKBONE_MODEL_ID
    hidden_size: int = 768
    layer_idx: int = -1
    apply_spec_augment: bool | None = False
    layerdrop: float | None = 0.0


class DistillModel(nn.Module):
    """Backbone plus optional projection head for chunk-level distillation."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        layer_idx: int = -1,
        target_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.layer_idx = layer_idx
        self.hidden_size = int(backbone.config.hidden_size)
        self.num_hidden_layers = int(getattr(backbone.config, "num_hidden_layers", 0))
        self.head = nn.Linear(self.hidden_size, target_dim) if target_dim is not None else None

    def encode(
        self,
        audio: torch.Tensor,
        *,
        layer: int | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if audio.dim() != 3:
            raise ValueError(f"expected audio shaped (B, N_chunks, T), got {tuple(audio.shape)}")

        batch, chunks, samples = audio.shape
        flat_audio = audio.reshape(batch * chunks, samples)
        flat_mask = None if attention_mask is None else attention_mask.reshape(batch * chunks, samples)

        outputs = self.backbone(
            input_values=flat_audio,
            attention_mask=flat_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        hidden = hidden_states[self.layer_idx if layer is None else layer]
        pooled = hidden.mean(dim=1)
        return pooled.reshape(batch, chunks, self.hidden_size)

    def forward(
        self,
        audio: torch.Tensor,
        *,
        layer: int | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = self.encode(audio, layer=layer, attention_mask=attention_mask)
        if self.head is None:
            return encoded
        return self.head(encoded)


def build_backbone(cfg: BackboneConfig, *, target_dim: int | None = None) -> DistillModel:
    backbone = AutoModel.from_pretrained(cfg.model_id, **hf_token_kwargs())
    if cfg.apply_spec_augment is not None and hasattr(backbone.config, "apply_spec_augment"):
        backbone.config.apply_spec_augment = bool(cfg.apply_spec_augment)
    if cfg.layerdrop is not None and hasattr(backbone.config, "layerdrop"):
        backbone.config.layerdrop = float(cfg.layerdrop)
    model_hidden = int(backbone.config.hidden_size)
    if model_hidden != cfg.hidden_size:
        raise ValueError(
            f"backbone hidden_size mismatch for {cfg.model_id}: config={cfg.hidden_size} model={model_hidden}"
        )
    return DistillModel(backbone, layer_idx=cfg.layer_idx, target_dim=target_dim)


def load_backbone(
    path: str | Path,
    *,
    model_id: str = DEFAULT_BACKBONE_MODEL_ID,
    layer_idx: int = -1,
) -> DistillModel:
    checkpoint = torch.load(Path(path), map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    model = DistillModel(AutoModel.from_pretrained(model_id, **hf_token_kwargs()), layer_idx=layer_idx)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


__all__ = [
    "BackboneConfig",
    "DEFAULT_BACKBONE_MODEL_ID",
    "DistillModel",
    "build_backbone",
    "load_backbone",
]