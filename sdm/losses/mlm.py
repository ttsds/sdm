"""Span-aware MLM masking for NeuCodec token sequences.

We use a 15% masking budget by default. Of the masked positions, 80% are
replaced with [MASK], 10% with a random FSQ token, 10% kept unchanged — the
classic BERT recipe. Special tokens (PAD/CLS/SEP/MASK) are never selected.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sdm.data.neucodec_dataset import MASK_ID, NUM_SPECIAL, PAD_ID


@dataclass
class MLMConfig:
    mask_prob: float = 0.15
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1


def mask_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    vocab_size: int,
    cfg: MLMConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (masked_input_ids, labels). Labels are -100 where loss is ignored."""
    labels = input_ids.clone()
    masked = input_ids.clone()

    eligible = (input_ids >= NUM_SPECIAL) & attention_mask.bool()
    rand = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    selected = (rand < cfg.mask_prob) & eligible
    labels[~selected] = -100

    rand2 = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    replace_with_mask = selected & (rand2 < cfg.mask_replace_prob)
    replace_with_random = (
        selected
        & (rand2 >= cfg.mask_replace_prob)
        & (rand2 < cfg.mask_replace_prob + cfg.random_replace_prob)
    )

    masked[replace_with_mask] = MASK_ID
    if replace_with_random.any():
        rand_tokens = torch.randint(
            low=NUM_SPECIAL,
            high=vocab_size,
            size=input_ids.shape,
            device=input_ids.device,
            generator=generator,
        )
        masked[replace_with_random] = rand_tokens[replace_with_random]

    # PAD positions never participate.
    masked[input_ids == PAD_ID] = PAD_ID
    labels[input_ids == PAD_ID] = -100
    return masked, labels
