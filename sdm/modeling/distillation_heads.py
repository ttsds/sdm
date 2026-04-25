"""Distillation heads that map sdm hidden states onto a teacher target.

There are two target families:

- **Sequence targets** (e.g. HuBERT layer 7 at 50 Hz): per-timestep 1:1 with
  sdm tokens; the head is a Linear over the hidden axis.
- **Pooled targets** (e.g. Whisper encoder mean over time): one vector per
  utterance; the head mean-pools the masked hidden states then projects.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class HeadSpec:
    """Description of a distillation target."""

    name: str
    target_dim: int
    pooled: bool = False  # True -> one vector per utterance; False -> per-timestep
    target_frame_rate_hz: float = 50.0  # used only when pooled=False


class DistillationHead(nn.Module):
    def __init__(self, hidden_size: int, spec: HeadSpec) -> None:
        super().__init__()
        self.spec = spec
        self.proj = nn.Linear(hidden_size, spec.target_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.spec.pooled:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (hidden_states * mask).sum(dim=1) / denom
            return self.proj(pooled)
        return self.proj(hidden_states)


class DistillationModel(nn.Module):
    """Backbone + dict of teacher heads. Backbone forward returns hidden states
    at the configured layer; each head produces an aligned prediction.

    The backbone is expected to be a HuggingFace `DebertaV2ForMaskedLM`. We
    delegate to its `.deberta` encoder so we don't carry the MLM lm_head into
    finetuning.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head_specs: list[HeadSpec],
        layer_idx: int = -1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.layer_idx = layer_idx
        hidden = backbone.config.hidden_size
        self.heads = nn.ModuleDict(
            {spec.name: DistillationHead(hidden, spec) for spec in head_specs}
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoder = getattr(self.backbone, "deberta", self.backbone)
        out = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = out.hidden_states  # tuple of (B, T, H)
        return hidden_states[self.layer_idx]

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        h = self.encode(input_ids, attention_mask)
        return {name: head(h, attention_mask) for name, head in self.heads.items()}


def distillation_loss(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    target_lengths: dict[str, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sum of per-teacher MSE losses, mean-reduced over valid frames.

    Sequence targets must already be temporally aligned to the model's tokens
    (length T_model). For pooled targets, predictions are (B, D), targets are
    (B, D). target_lengths/attention_mask only matter for sequence heads.
    """
    weights = weights or {}
    losses: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, float] = {}
    for name, pred in preds.items():
        target = targets[name]
        if pred.dim() == 3:  # sequence head
            assert attention_mask is not None
            mask = attention_mask.to(pred.dtype).unsqueeze(-1)
            sq = (pred - target) ** 2 * mask
            denom = mask.sum().clamp_min(1.0) * pred.shape[-1]
            loss = sq.sum() / denom
        else:  # pooled head
            loss = torch.mean((pred - target) ** 2)
        w = weights.get(name, 1.0)
        losses[name] = w * loss
        diagnostics[name] = float(loss.detach())
    total = torch.stack(list(losses.values())).sum()
    return total, diagnostics


__all__ = [
    "DistillationHead",
    "DistillationModel",
    "HeadSpec",
    "distillation_loss",
]
