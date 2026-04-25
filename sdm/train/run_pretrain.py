"""YAML-config entrypoint for pretraining. Used by the TPU launch script."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from sdm.data.neucodec_dataset import NeucodecConfig
from sdm.losses.mlm import MLMConfig
from sdm.losses.wristband_gaussian import GaussianLossConfig
from sdm.modeling.deberta_neucodec import SdmConfig
from sdm.train import wandb_utils
from sdm.train.pretrain import TrainConfig, train


def load_config(path: str | Path) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text())
    model = SdmConfig(**raw["model"])
    data = NeucodecConfig(**raw["data"])
    mlm = MLMConfig(**raw["mlm"])
    train_kwargs = dict(raw["train"])
    if "gaussian_loss" in train_kwargs:
        train_kwargs["gaussian_loss"] = GaussianLossConfig(**train_kwargs["gaussian_loss"])
    if "wandb" in train_kwargs:
        train_kwargs["wandb"] = wandb_utils.WandbConfig(**train_kwargs["wandb"])
    return TrainConfig(model=model, data=data, mlm=mlm, **train_kwargs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    train(cfg, synthetic=args.synthetic)


if __name__ == "__main__":
    main()
