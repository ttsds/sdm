from __future__ import annotations

from sdm.train.run_distill import load_config


def test_load_post_pivot_distill_config():
    cfg = load_config("configs/finetune_xlsr.yaml")

    assert cfg.experiment == "sdm-xlsr"
    assert cfg.backbone.model_id == "utter-project/mHuBERT-147"
    assert cfg.teacher.kind == "hf_ssl"
    assert cfg.teacher.target_dim == 1024
    assert cfg.data.repo_id == "amphion/Emilia-Dataset"
    assert cfg.train.batch_size == 16