"""Self-contained port of fairseq's wav2vec2 encoder.

Architecture is taken directly from
``fairseq/fairseq/models/wav2vec/wav2vec2.py`` (MIT-licensed, Facebook /
fairseq authors). Only the pieces needed for *encoder forward* are vendored:

- :class:`ConvFeatureExtractionModel` (fp32 GroupNorm / LayerNorm)
- :class:`TransformerSentenceEncoderLayer` (pre-LN / post-LN paths)
- :class:`TransformerEncoder` with an optional positional convolution
- :class:`Wav2Vec2Model` exposing a HuggingFace-style ``forward`` interface
  (``input_values``, ``output_hidden_states``, returns an object with
  ``.hidden_states``) so :class:`sdm.modeling.distill_model.DistillModel` can
  consume it without changes.

What is intentionally *not* ported: GumbelVectorQuantizer, target / mask
sampling, contrastive loss, conformer / adapter layer types, layerdrop logic
that depends on numpy host RNG, FSDP wrappers, gradient-checkpoint wrappers.
For distillation we only need a deterministic forward that returns the per-
layer hidden states.

A separate :func:`load_xlsr_from_hf` helper converts a HuggingFace
``Wav2Vec2Model`` state dict (e.g. ``facebook/wav2vec2-xls-r-300m``) into
this layout so we keep the same checkpoint source we've been using.

Why this exists at all: HF's wav2vec2 / hubert run their LayerNorm and
GroupNorm in the surrounding autocast dtype. On bf16 XLA / TPU the
GroupNorm-over-512-channels-with-1-group pattern in ``feat_extract_norm=
"group"`` and the bare LayerNorm in the positional-conv block both produce
NaN gradients. fairseq deliberately wraps both in fp32 (``Fp32GroupNorm``,
``Fp32LayerNorm``) precisely to avoid this; keeping that contract is the
main reason the port matters here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper modules (verbatim from fairseq, with minor fp32-LN addition).
# ---------------------------------------------------------------------------


class Fp32GroupNorm(nn.GroupNorm):
    """``nn.GroupNorm`` that runs internally in fp32 and casts back."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class Fp32LayerNorm(nn.LayerNorm):
    """``nn.LayerNorm`` that runs internally in fp32 and casts back."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class SamePad(nn.Module):
    def __init__(self, kernel_size: int, causal: bool = False) -> None:
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class TransposeLast(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(-2, -1)


# ---------------------------------------------------------------------------
# Feature extractor: 7 conv blocks, default XLS-R config uses layer_norm mode.
# ---------------------------------------------------------------------------


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "layer_norm",
        conv_bias: bool = True,
    ) -> None:
        super().__init__()
        assert mode in {"default", "layer_norm"}

        def block(
            n_in: int,
            n_out: int,
            k: int,
            stride: int,
            *,
            is_layer_norm: bool,
            is_group_norm: bool,
            conv_bias: bool,
        ) -> nn.Module:
            assert not (is_layer_norm and is_group_norm)
            conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
            nn.init.kaiming_normal_(conv.weight)

            if is_layer_norm:
                return nn.Sequential(
                    conv,
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(n_out, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            if is_group_norm:
                return nn.Sequential(
                    conv,
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(n_out, n_out, affine=True),
                    nn.GELU(),
                )
            return nn.Sequential(conv, nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, (dim, k, stride) in enumerate(conv_layers):
            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T) -> (B, 1, T)
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x


# ---------------------------------------------------------------------------
# Positional convolution (weight-normalised by default in fairseq; we strip
# the parametrization at construction time so the bf16 backward divide-by-
# norm doesn't NaN on XLA).
# ---------------------------------------------------------------------------


def make_conv_pos(e: int, k: int, g: int) -> nn.Sequential:
    pos_conv = nn.Conv1d(e, e, kernel_size=k, padding=k // 2, groups=g)
    std = math.sqrt(4.0 / (k * e))
    nn.init.normal_(pos_conv.weight, mean=0.0, std=std)
    nn.init.constant_(pos_conv.bias, 0.0)
    pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
    seq = nn.Sequential(pos_conv, SamePad(k), nn.GELU())
    _strip_weight_norm(seq[0])
    return seq


def _strip_weight_norm(conv: nn.Conv1d) -> None:
    """Bake out ``weight_norm`` into a plain ``Conv1d.weight``."""
    try:
        from torch.nn.utils import parametrize  # type: ignore[attr-defined]

        if parametrize.is_parametrized(conv, "weight"):
            parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
            return
    except Exception:
        pass
    if hasattr(conv, "weight_g") and hasattr(conv, "weight_v"):
        nn.utils.remove_weight_norm(conv, name="weight")


# ---------------------------------------------------------------------------
# Transformer encoder layer / stack. Uses ``nn.MultiheadAttention`` so the
# checkpoint converter can splat HF's separate q/k/v projections into the
# combined ``in_proj_weight`` tensor.
# ---------------------------------------------------------------------------


class TransformerSentenceEncoderLayer(nn.Module):
    """fairseq's wav2vec2 transformer layer (pre-LN or post-LN)."""

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 12,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        layer_norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.layer_norm_first = layer_norm_first

        if activation_fn != "gelu":
            raise ValueError(f"unsupported activation_fn for the port: {activation_fn!r}")
        self.activation_fn = F.gelu

        self.self_attn = nn.MultiheadAttention(
            embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            batch_first=False,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, _ = self.self_attn(
                x, x, x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, _ = self.self_attn(
                x, x, x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
            )
            x = self.dropout1(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, cfg: "Wav2Vec2Config") -> None:
        super().__init__()
        self.dropout = cfg.dropout
        self.embedding_dim = cfg.encoder_embed_dim
        self.layer_norm_first = cfg.layer_norm_first

        self.pos_conv = make_conv_pos(
            cfg.encoder_embed_dim,
            cfg.conv_pos,
            cfg.conv_pos_groups,
        )
        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=cfg.encoder_embed_dim,
                    ffn_embedding_dim=cfg.encoder_ffn_embed_dim,
                    num_attention_heads=cfg.encoder_attention_heads,
                    dropout=cfg.dropout,
                    attention_dropout=cfg.attention_dropout,
                    activation_dropout=cfg.activation_dropout,
                    activation_fn=cfg.activation_fn,
                    layer_norm_first=cfg.layer_norm_first,
                )
                for _ in range(cfg.encoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        x_conv = self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C for nn.MultiheadAttention with batch_first=False.
        x = x.transpose(0, 1)

        hidden_states: list[torch.Tensor] = []
        if output_hidden_states:
            hidden_states.append(x.transpose(0, 1))

        for layer in self.layers:
            x = layer(x, self_attn_padding_mask=padding_mask)
            if output_hidden_states:
                hidden_states.append(x.transpose(0, 1))

        x = x.transpose(0, 1)
        if self.layer_norm_first:
            x = self.layer_norm(x)
        if output_hidden_states:
            hidden_states[-1] = x
        return x, tuple(hidden_states)


# ---------------------------------------------------------------------------
# Top-level model + config.
# ---------------------------------------------------------------------------


@dataclass
class Wav2Vec2Config:
    """Subset of fairseq's ``Wav2Vec2Config`` covering encoder forward only.

    Defaults are XLS-R 300M (24 layers, 1024-dim, layer_norm extractor mode,
    layer_norm_first encoder, GELU FFNs, ``conv_pos=128``).
    """

    extractor_mode: str = "layer_norm"
    encoder_layers: int = 24
    encoder_embed_dim: int = 1024
    encoder_ffn_embed_dim: int = 4096
    encoder_attention_heads: int = 16
    activation_fn: str = "gelu"
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    layer_norm_first: bool = True
    conv_feature_layers: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ]
    )
    conv_bias: bool = True
    conv_pos: int = 128
    conv_pos_groups: int = 16
    feature_grad_mult: float = 1.0


class _Wav2Vec2Output:
    """Small object mirroring HF's ``BaseModelOutput``."""

    def __init__(self, last_hidden_state: torch.Tensor, hidden_states: tuple[torch.Tensor, ...]):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class _Wav2Vec2RuntimeConfig:
    """HF-compatible config object exposed via ``model.config``."""

    def __init__(self, cfg: Wav2Vec2Config):
        self.hidden_size = cfg.encoder_embed_dim
        self.num_hidden_layers = cfg.encoder_layers


class Wav2Vec2Model(nn.Module):
    """Fairseq-faithful encoder wrapped in the HF-style forward interface."""

    def __init__(self, cfg: Wav2Vec2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.config = _Wav2Vec2RuntimeConfig(cfg)
        self.feature_extractor = ConvFeatureExtractionModel(
            cfg.conv_feature_layers,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )
        extractor_dim = cfg.conv_feature_layers[-1][0]
        self.layer_norm = Fp32LayerNorm(extractor_dim, elementwise_affine=True)
        self.post_extract_proj = (
            nn.Linear(extractor_dim, cfg.encoder_embed_dim)
            if extractor_dim != cfg.encoder_embed_dim
            else None
        )
        self.encoder = TransformerEncoder(cfg)
        self.feature_grad_mult = cfg.feature_grad_mult

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **_: object,
    ) -> _Wav2Vec2Output:
        del attention_mask  # not consumed; we don't pass padding through here
        # input_values: (B, T) -> feature extractor -> (B, C, N) -> (B, N, C).
        feats = self.feature_extractor(input_values)
        if self.feature_grad_mult != 1.0 and feats.requires_grad:
            feats = _GradMultiply.apply(feats, self.feature_grad_mult)
        feats = feats.transpose(1, 2)
        feats = self.layer_norm(feats)
        if self.post_extract_proj is not None:
            feats = self.post_extract_proj(feats)
        last, hiddens = self.encoder(feats, output_hidden_states=output_hidden_states)
        return _Wav2Vec2Output(last_hidden_state=last, hidden_states=hiddens)


class _GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:  # type: ignore[override]
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # type: ignore[override]
        return grad * ctx.scale, None


# ---------------------------------------------------------------------------
# HuggingFace -> fairseq state-dict converter.
# ---------------------------------------------------------------------------


def _hf_extractor_subkey(layer_idx: int, mode: str, attr: str) -> str:
    """Map a fairseq feature-extractor parameter to its HF counterpart.

    Fairseq stores each block as ``nn.Sequential(conv, dropout, [LN], gelu)``
    so the conv weights live under ``conv_layers.{i}.0.{weight,bias}`` and the
    layer-norm under ``conv_layers.{i}.2.1.{weight,bias}``. HF flattens these
    into ``feature_extractor.conv_layers.{i}.{conv,layer_norm}.{weight,bias}``.
    """
    if attr in {"conv.weight", "conv.bias"}:
        sub = attr.split(".", 1)[1]
        return f"feature_extractor.conv_layers.{layer_idx}.0.{sub}"
    if attr in {"layer_norm.weight", "layer_norm.bias"}:
        sub = attr.split(".", 1)[1]
        if mode == "layer_norm":
            return f"feature_extractor.conv_layers.{layer_idx}.2.1.{sub}"
        # mode == "default": GroupNorm only on layer 0.
        return f"feature_extractor.conv_layers.{layer_idx}.2.{sub}"
    raise KeyError(attr)


@torch.no_grad()
def load_xlsr_from_hf(model_id: str = "facebook/wav2vec2-xls-r-300m") -> Wav2Vec2Model:
    """Build a fairseq-port :class:`Wav2Vec2Model` and load HF XLS-R weights.

    The HF checkpoint already mirrors fairseq's architecture (it's a direct
    port the other direction), so the conversion is a deterministic key
    rename. We splat HF's separate ``q_proj``/``k_proj``/``v_proj`` into the
    combined ``in_proj_weight`` tensor expected by ``nn.MultiheadAttention``.
    """
    from transformers import AutoConfig, AutoModel  # type: ignore

    from sdm.dotenv import hf_token_kwargs

    hf_cfg = AutoConfig.from_pretrained(model_id, **hf_token_kwargs())
    cfg = Wav2Vec2Config(
        extractor_mode="layer_norm" if hf_cfg.feat_extract_norm == "layer" else "default",
        encoder_layers=hf_cfg.num_hidden_layers,
        encoder_embed_dim=hf_cfg.hidden_size,
        encoder_ffn_embed_dim=hf_cfg.intermediate_size,
        encoder_attention_heads=hf_cfg.num_attention_heads,
        activation_fn="gelu",
        layer_norm_first=bool(hf_cfg.do_stable_layer_norm),
        conv_feature_layers=[
            (int(d), int(k), int(s))
            for d, k, s in zip(hf_cfg.conv_dim, hf_cfg.conv_kernel, hf_cfg.conv_stride)
        ],
        conv_bias=bool(hf_cfg.conv_bias),
        conv_pos=int(hf_cfg.num_conv_pos_embeddings),
        conv_pos_groups=int(hf_cfg.num_conv_pos_embedding_groups),
    )
    model = Wav2Vec2Model(cfg)

    hf_model = AutoModel.from_pretrained(model_id, **hf_token_kwargs())
    src = hf_model.state_dict()

    new_state: dict[str, torch.Tensor] = {}

    # Feature extractor.
    for i, _ in enumerate(cfg.conv_feature_layers):
        new_state[f"feature_extractor.conv_layers.{i}.0.weight"] = src[
            f"feature_extractor.conv_layers.{i}.conv.weight"
        ]
        if cfg.conv_bias and f"feature_extractor.conv_layers.{i}.conv.bias" in src:
            new_state[f"feature_extractor.conv_layers.{i}.0.bias"] = src[
                f"feature_extractor.conv_layers.{i}.conv.bias"
            ]
        if cfg.extractor_mode == "layer_norm":
            new_state[f"feature_extractor.conv_layers.{i}.2.1.weight"] = src[
                f"feature_extractor.conv_layers.{i}.layer_norm.weight"
            ]
            new_state[f"feature_extractor.conv_layers.{i}.2.1.bias"] = src[
                f"feature_extractor.conv_layers.{i}.layer_norm.bias"
            ]
        elif i == 0:  # default mode: GroupNorm only on first block.
            new_state[f"feature_extractor.conv_layers.{i}.2.weight"] = src[
                f"feature_extractor.conv_layers.{i}.layer_norm.weight"
            ]
            new_state[f"feature_extractor.conv_layers.{i}.2.bias"] = src[
                f"feature_extractor.conv_layers.{i}.layer_norm.bias"
            ]

    # Post-extractor projection + layer norm (HF: feature_projection.*).
    new_state["layer_norm.weight"] = src["feature_projection.layer_norm.weight"]
    new_state["layer_norm.bias"] = src["feature_projection.layer_norm.bias"]
    if model.post_extract_proj is not None:
        new_state["post_extract_proj.weight"] = src["feature_projection.projection.weight"]
        new_state["post_extract_proj.bias"] = src["feature_projection.projection.bias"]

    # Positional convolution. HF uses weight_norm too; baking it produces a
    # plain ``conv.weight`` we can copy directly.
    pos_conv = hf_model.encoder.pos_conv_embed.conv
    _strip_weight_norm(pos_conv)
    new_state["encoder.pos_conv.0.weight"] = pos_conv.weight.detach().clone()
    new_state["encoder.pos_conv.0.bias"] = pos_conv.bias.detach().clone()

    # Top-of-encoder layer norm (post-LN final norm in fairseq).
    new_state["encoder.layer_norm.weight"] = src["encoder.layer_norm.weight"]
    new_state["encoder.layer_norm.bias"] = src["encoder.layer_norm.bias"]

    # Encoder layers.
    for i in range(cfg.encoder_layers):
        # MultiheadAttention combined in_proj.
        q_w = src[f"encoder.layers.{i}.attention.q_proj.weight"]
        k_w = src[f"encoder.layers.{i}.attention.k_proj.weight"]
        v_w = src[f"encoder.layers.{i}.attention.v_proj.weight"]
        q_b = src[f"encoder.layers.{i}.attention.q_proj.bias"]
        k_b = src[f"encoder.layers.{i}.attention.k_proj.bias"]
        v_b = src[f"encoder.layers.{i}.attention.v_proj.bias"]
        new_state[f"encoder.layers.{i}.self_attn.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        new_state[f"encoder.layers.{i}.self_attn.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0)
        new_state[f"encoder.layers.{i}.self_attn.out_proj.weight"] = src[
            f"encoder.layers.{i}.attention.out_proj.weight"
        ]
        new_state[f"encoder.layers.{i}.self_attn.out_proj.bias"] = src[
            f"encoder.layers.{i}.attention.out_proj.bias"
        ]

        # Layer norms.
        new_state[f"encoder.layers.{i}.self_attn_layer_norm.weight"] = src[
            f"encoder.layers.{i}.layer_norm.weight"
        ]
        new_state[f"encoder.layers.{i}.self_attn_layer_norm.bias"] = src[
            f"encoder.layers.{i}.layer_norm.bias"
        ]
        new_state[f"encoder.layers.{i}.final_layer_norm.weight"] = src[
            f"encoder.layers.{i}.final_layer_norm.weight"
        ]
        new_state[f"encoder.layers.{i}.final_layer_norm.bias"] = src[
            f"encoder.layers.{i}.final_layer_norm.bias"
        ]

        # Feed-forward.
        new_state[f"encoder.layers.{i}.fc1.weight"] = src[
            f"encoder.layers.{i}.feed_forward.intermediate_dense.weight"
        ]
        new_state[f"encoder.layers.{i}.fc1.bias"] = src[
            f"encoder.layers.{i}.feed_forward.intermediate_dense.bias"
        ]
        new_state[f"encoder.layers.{i}.fc2.weight"] = src[
            f"encoder.layers.{i}.feed_forward.output_dense.weight"
        ]
        new_state[f"encoder.layers.{i}.fc2.bias"] = src[
            f"encoder.layers.{i}.feed_forward.output_dense.bias"
        ]

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        raise RuntimeError(f"missing keys after HF->fairseq port: {missing}")
    if unexpected:
        raise RuntimeError(f"unexpected keys after HF->fairseq port: {unexpected}")
    return model


__all__ = [
    "ConvFeatureExtractionModel",
    "Fp32GroupNorm",
    "Fp32LayerNorm",
    "TransformerEncoder",
    "TransformerSentenceEncoderLayer",
    "Wav2Vec2Config",
    "Wav2Vec2Model",
    "load_xlsr_from_hf",
]
