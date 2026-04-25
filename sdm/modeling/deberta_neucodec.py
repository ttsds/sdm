"""DeBERTa-v3 backbone configured for NeuCodec FSQ tokens.

We use HuggingFace's `DebertaV2ForMaskedLM` directly. The only NeuCodec-specific
choices are the vocab size (FSQ codebook + 4 special tokens) and the position
embedding range (sized for ~30 s utterances at ~50 Hz).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from transformers import DebertaV2Config, DebertaV2ForMaskedLM

from sdm.data.neucodec_dataset import (
    CLS_ID,
    MASK_ID,
    NUM_SPECIAL,
    PAD_ID,
    SEP_ID,
)


@dataclass
class SdmConfig:
    """Subset of DebertaV2Config exposed at sdm's level."""

    fsq_vocab_size: int = 65536  # placeholder; confirmed empirically per dataset
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    relative_attention: bool = True
    position_buckets: int = 256
    pos_att_type: list[str] = field(default_factory=lambda: ["p2c", "c2p"])
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-7

    @property
    def vocab_size(self) -> int:
        return self.fsq_vocab_size + NUM_SPECIAL


def build_deberta_config(cfg: SdmConfig) -> DebertaV2Config:
    return DebertaV2Config(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        max_position_embeddings=cfg.max_position_embeddings,
        relative_attention=cfg.relative_attention,
        position_buckets=cfg.position_buckets,
        pos_att_type=cfg.pos_att_type,
        hidden_dropout_prob=cfg.hidden_dropout_prob,
        attention_probs_dropout_prob=cfg.attention_probs_dropout_prob,
        layer_norm_eps=cfg.layer_norm_eps,
        pad_token_id=PAD_ID,
        bos_token_id=CLS_ID,
        eos_token_id=SEP_ID,
    )


def build_model(cfg: SdmConfig) -> DebertaV2ForMaskedLM:
    return DebertaV2ForMaskedLM(build_deberta_config(cfg))


SDM_SMALL = SdmConfig(
    hidden_size=384,
    num_hidden_layers=6,
    num_attention_heads=6,
    intermediate_size=1536,
)

SDM_BASE = SdmConfig()  # 12L/768H — DeBERTa-v3-base footprint


__all__ = [
    "CLS_ID",
    "MASK_ID",
    "PAD_ID",
    "SEP_ID",
    "SDM_BASE",
    "SDM_SMALL",
    "SdmConfig",
    "build_deberta_config",
    "build_model",
]
