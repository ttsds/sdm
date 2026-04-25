import torch

from sdm.data.neucodec_dataset import MASK_ID, NUM_SPECIAL, PAD_ID, codes_to_input_ids
from sdm.losses.mlm import MLMConfig, mask_tokens


def test_mask_tokens_respects_special_tokens():
    ids, attn = codes_to_input_ids([10, 20, 30, 40], max_length=8)
    batch_ids = ids.unsqueeze(0)
    batch_attn = attn.unsqueeze(0)
    cfg = MLMConfig(mask_prob=1.0, mask_replace_prob=1.0, random_replace_prob=0.0)
    g = torch.Generator().manual_seed(0)
    masked, labels = mask_tokens(batch_ids, batch_attn, vocab_size=128, cfg=cfg, generator=g)

    # Only positions that originally held FSQ codes (>= NUM_SPECIAL) get masked.
    pad_positions = batch_ids == PAD_ID
    assert (masked[pad_positions] == PAD_ID).all()
    assert (labels[pad_positions] == -100).all()

    fsq_positions = batch_ids >= NUM_SPECIAL
    assert (masked[fsq_positions] == MASK_ID).all()
    assert (labels[fsq_positions] == batch_ids[fsq_positions]).all()


def test_mask_tokens_zero_prob_yields_no_loss_targets():
    ids, attn = codes_to_input_ids([10, 20, 30], max_length=8)
    batch_ids = ids.unsqueeze(0)
    batch_attn = attn.unsqueeze(0)
    cfg = MLMConfig(mask_prob=0.0)
    masked, labels = mask_tokens(batch_ids, batch_attn, vocab_size=128, cfg=cfg)
    assert (labels == -100).all()
    assert (masked == batch_ids).all()
