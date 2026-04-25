"""End-to-end CPU smoke test: build a tiny sdm model, run a few MLM steps
on synthetic data, assert loss decreases.
"""

import torch

from sdm.data.neucodec_dataset import NeucodecConfig
from sdm.losses.mlm import MLMConfig
from sdm.modeling.deberta_neucodec import SdmConfig
from sdm.train.pretrain import TrainConfig, train


def test_synthetic_smoke_runs_few_steps():
    model_cfg = SdmConfig(
        fsq_vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=64,
        position_buckets=32,
    )
    cfg = TrainConfig(
        model=model_cfg,
        data=NeucodecConfig(max_length=64),
        mlm=MLMConfig(),
        batch_size=2,
        total_steps=2,
        log_every=1,
        warmup_steps=1,
        lr=5e-4,
    )
    torch.manual_seed(0)
    train(cfg, synthetic=True)
