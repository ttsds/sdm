"""mHuBERT backbone wrapper used by the post-pivot distillation pipeline.

The distillation configs operate on raw audio chunk tensors rather than
NeuCodec token ids. This module provides a thin wrapper around a HuggingFace
audio backbone so both training and probing can share the same chunk-level
latent extraction path.
"""

from __future__ import annotations

import math
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
    # ``"hf"`` loads ``AutoModel.from_pretrained(model_id)``. ``"tiny"`` builds
    # an in-tree minimal Conv1d + Transformer encoder used for debugging.
    kind: str = "hf"
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    frame_kernel: int = 400
    frame_stride: int = 320
    sample_rate: int = 16000


class _TinyBackboneOutput:
    def __init__(self, last_hidden_state: torch.Tensor, hidden_states: tuple[torch.Tensor, ...]):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class _TinyBackboneConfig:
    def __init__(self, hidden_size: int, num_hidden_layers: int):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers


class TinyAudioBackbone(nn.Module):
    """Minimal raw-audio transformer used to debug the distillation pipeline.

    Mimics the subset of the HuggingFace ``AutoModel`` interface used by
    :class:`DistillModel`: accepts ``input_values=(B, T)`` plus
    ``output_hidden_states=True`` / ``return_dict=True`` and returns an object
    with ``.hidden_states`` (tuple of length ``num_hidden_layers + 1``).

    The frontend is a single ``Conv1d(kernel=400, stride=320)`` so 16 kHz
    audio downsamples to ~50 Hz frames, matching wav2vec2/HuBERT output rate.
    Sinusoidal positional embeddings + standard pre-LN transformer layers keep
    the activation graph free of weight_norm, group norm, and other XLA
    footguns.
    """

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.config = _TinyBackboneConfig(cfg.hidden_size, cfg.num_hidden_layers)
        self.frontend = nn.Conv1d(
            in_channels=1,
            out_channels=cfg.hidden_size,
            kernel_size=cfg.frame_kernel,
            stride=cfg.frame_stride,
            padding=cfg.frame_kernel // 2,
        )
        self.frontend_norm = nn.LayerNorm(cfg.hidden_size)
        self.frontend_act = nn.GELU()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_size,
            nhead=cfg.num_attention_heads,
            dim_feedforward=cfg.intermediate_size,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.ModuleList(
            [encoder_layer.__class__(
                d_model=cfg.hidden_size,
                nhead=cfg.num_attention_heads,
                dim_feedforward=cfg.intermediate_size,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ) for _ in range(cfg.num_hidden_layers)]
        )
        self.final_norm = nn.LayerNorm(cfg.hidden_size)
        self._init_weights()

    def _init_weights(self) -> None:
        # Small std init keeps activations bounded at step 0; matches what
        # most audio transformers ship with.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _sinusoidal(seq_len: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(seq_len, dim, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.to(dtype)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **_: object,
    ) -> _TinyBackboneOutput:
        del attention_mask  # not used by the tiny model
        # input_values: (B, T) raw audio.
        x = input_values.unsqueeze(1)  # (B, 1, T)
        x = self.frontend(x)  # (B, D, N)
        x = x.transpose(1, 2)  # (B, N, D)
        x = self.frontend_norm(x)
        x = self.frontend_act(x)
        x = x + self._sinusoidal(x.shape[1], x.shape[2], x.device, x.dtype)

        hidden_states: list[torch.Tensor] = [x] if output_hidden_states else []
        for layer in self.layers:
            x = layer(x)
            if output_hidden_states:
                hidden_states.append(x)
        last = self.final_norm(x)
        if output_hidden_states:
            hidden_states[-1] = last
        return _TinyBackboneOutput(
            last_hidden_state=last,
            hidden_states=tuple(hidden_states) if output_hidden_states else (),
        )


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


def _strip_pos_conv_weight_norm(backbone: nn.Module) -> None:
    """Bake out ``weight_norm`` on the positional convolution.

    HuBERT/Wav2Vec2 wrap ``pos_conv_embed.conv`` in ``weight_norm`` which
    reparameterises ``weight = weight_v * weight_g / ||weight_v||``. The
    backward pass divides by ``||weight_v||``; on bf16 XLA this routinely
    produces NaN gradients (the well-known XLA Wav2Vec2 NaN). Folding the
    parametrization into a plain ``Conv1d.weight`` is numerically a no-op at
    load time and removes the offending divide entirely.
    """
    encoder = getattr(backbone, "encoder", None)
    pos = getattr(encoder, "pos_conv_embed", None) if encoder is not None else None
    conv = getattr(pos, "conv", None) if pos is not None else None
    if conv is None:
        return
    # New parametrizations API.
    try:
        from torch.nn.utils import parametrize  # type: ignore[attr-defined]

        if parametrize.is_parametrized(conv, "weight"):
            parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
            return
    except Exception:
        pass
    # Legacy weight_norm API.
    if hasattr(conv, "weight_g") and hasattr(conv, "weight_v"):
        torch.nn.utils.remove_weight_norm(conv, name="weight")


def build_backbone(cfg: BackboneConfig, *, target_dim: int | None = None) -> DistillModel:
    if cfg.kind == "tiny":
        backbone = TinyAudioBackbone(cfg)
        return DistillModel(backbone, layer_idx=cfg.layer_idx, target_dim=target_dim)
    if cfg.kind == "fairseq_w2v2":
        from sdm.modeling.wav2vec2_fairseq import load_xlsr_from_hf

        backbone = load_xlsr_from_hf(cfg.model_id)
        model_hidden = int(backbone.config.hidden_size)
        if model_hidden != cfg.hidden_size:
            raise ValueError(
                f"backbone hidden_size mismatch for {cfg.model_id}: config={cfg.hidden_size} model={model_hidden}"
            )
        return DistillModel(backbone, layer_idx=cfg.layer_idx, target_dim=target_dim)
    if cfg.kind != "hf":
        raise ValueError(f"unknown backbone kind: {cfg.kind!r}")
    backbone = AutoModel.from_pretrained(cfg.model_id, **hf_token_kwargs())
    if cfg.apply_spec_augment is not None and hasattr(backbone.config, "apply_spec_augment"):
        backbone.config.apply_spec_augment = bool(cfg.apply_spec_augment)
    if cfg.layerdrop is not None and hasattr(backbone.config, "layerdrop"):
        backbone.config.layerdrop = float(cfg.layerdrop)
    _strip_pos_conv_weight_norm(backbone)
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
    backbone = AutoModel.from_pretrained(model_id, **hf_token_kwargs())
    _strip_pos_conv_weight_norm(backbone)
    model = DistillModel(backbone, layer_idx=layer_idx)
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