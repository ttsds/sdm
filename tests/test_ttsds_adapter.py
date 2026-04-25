"""Light tests for the TTSDS adapter. The full upstream `ttsds` import path is
behind the `teachers` extra and not present in the default dev env, so we
stick to the parts we can exercise without it.
"""

import numpy as np
import torch

from sdm.data.neucodec_dataset import codes_to_input_ids
from sdm.modeling.deberta_neucodec import SdmConfig, build_model
from sdm.modeling.distillation_heads import DistillationModel, HeadSpec


def test_distillation_model_inference_path():
    cfg = SdmConfig(
        fsq_vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        position_buckets=16,
    )
    backbone = build_model(cfg)
    spec = HeadSpec(name="hubert", target_dim=8, pooled=False)
    model = DistillationModel(backbone, [spec])

    codes = np.random.randint(0, cfg.fsq_vocab_size, size=20).tolist()
    input_ids, attn = codes_to_input_ids(codes, max_length=64)
    out = model(input_ids.unsqueeze(0), attn.unsqueeze(0))
    seq = out["hubert"].squeeze(0)
    valid = attn.bool()
    embedded = seq[valid].detach().numpy()
    assert embedded.ndim == 2
    assert embedded.shape[1] == 8
    assert embedded.shape[0] == int(attn.sum().item())


def test_pooled_head_inference_path():
    cfg = SdmConfig(
        fsq_vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        position_buckets=16,
    )
    backbone = build_model(cfg)
    spec = HeadSpec(name="whisper", target_dim=24, pooled=True)
    model = DistillationModel(backbone, [spec])

    input_ids = torch.randint(4, cfg.vocab_size, (1, 32))
    attn = torch.ones(1, 32, dtype=torch.long)
    out = model(input_ids, attn)
    assert out["whisper"].shape == (1, 24)
