from __future__ import annotations

from sdm.train.run_distill import load_config


def test_load_post_pivot_distill_config():
    cfg = load_config("configs/finetune_xlsr_fairseq.yaml")

    assert cfg.experiment == "sdm-xlsr-fairseq"
    assert cfg.backbone.kind == "fairseq_w2v2"
    assert cfg.backbone.model_id == "facebook/wav2vec2-xls-r-300m"
    assert cfg.teacher.kind == "hf_ssl"
    assert cfg.teacher.model_id == "utter-project/mHuBERT-147"
    assert cfg.teacher.target_dim == 768
    assert cfg.data.repo_id == "amphion/Emilia-Dataset"