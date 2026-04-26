"""Per-factor distillation finetune. One config per factor; each spec maps a
teacher name to its `HeadSpec` (target dim, pooled vs sequence).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from sdm.data.neucodec_dataset import NeucodecConfig
from sdm.data.teacher_dataset import (
    TeacherCacheConfig,
    TeacherShardDataset,
    collate_teacher_batch,
)
from sdm.dotenv import load_dotenv
from sdm.losses.wristband_gaussian import GaussianLossConfig, WristbandGaussianLoss
from sdm.modeling.deberta_neucodec import SdmConfig, build_model
from sdm.modeling.distillation_heads import (
    DistillationModel,
    HeadSpec,
    distillation_loss,
)
from sdm.train import io as ckpt_io
from sdm.train import preempt, wandb_utils, xla_utils


@dataclass
class FinetuneConfig:
    model: SdmConfig
    data: NeucodecConfig
    cache_dir: Path
    head_specs: list[HeadSpec]
    pretrained_ckpt: str | None = None
    layer_idx: int = -1
    batch_size: int = 8
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    total_steps: int = 50_000
    log_every: int = 50
    grad_accum: int = 1
    ckpt_dir: str | None = None
    ckpt_every: int = 5000
    seed: int = 0
    resume_from_latest: bool = True
    teacher_weights: dict[str, float] = field(default_factory=dict)
    gaussian_loss: GaussianLossConfig = field(default_factory=GaussianLossConfig)
    wandb: wandb_utils.WandbConfig = field(default_factory=wandb_utils.WandbConfig)


def _lr_at(step: int, cfg: FinetuneConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train(cfg: FinetuneConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = xla_utils.get_device(require_xla=xla_utils.xla_required())
    stop = preempt.install()

    backbone = build_model(cfg.model)
    if cfg.pretrained_ckpt:
        state = ckpt_io.load_state(cfg.pretrained_ckpt, map_location="cpu")
        backbone.load_state_dict(state["model"], strict=False)

    model = DistillationModel(backbone, cfg.head_specs, layer_idx=cfg.layer_idx).to(device)
    model.train()

    pooled_map = {s.name: s.pooled for s in cfg.head_specs}
    cache_cfg = TeacherCacheConfig(
        shard_dir=Path(cfg.cache_dir),
        teachers=[s.name for s in cfg.head_specs],
        max_length=cfg.data.max_length,
        pooled=pooled_map,
    )
    ds = TeacherShardDataset(cache_cfg)
    loader = DataLoader(ds, batch_size=cfg.batch_size, collate_fn=collate_teacher_batch)
    device_loader = xla_utils.loader_per_device(loader, device)

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=cfg.weight_decay
    )

    gaussian_fn: WristbandGaussianLoss | None = None
    if cfg.gaussian_loss.enabled:
        cal_shape = (
            (cfg.batch_size, cfg.model.hidden_size) if cfg.gaussian_loss.calibrate else None
        )
        gaussian_fn = WristbandGaussianLoss(
            beta=cfg.gaussian_loss.beta,
            lambda_rad=cfg.gaussian_loss.lambda_rad,
            lambda_mom=cfg.gaussian_loss.lambda_mom,
            moment=cfg.gaussian_loss.moment,
            calibration_shape=cal_shape,
        )

    start_step = 0
    if cfg.ckpt_dir and cfg.resume_from_latest:
        latest = ckpt_io.latest_checkpoint(cfg.ckpt_dir)
        if latest is not None:
            if xla_utils.is_master():
                print(f"resuming from {latest}")
            state = ckpt_io.load_state(latest, map_location="cpu")
            model_ok, reason = xla_utils.state_dict_is_finite(state["model"])
            if not model_ok:
                if xla_utils.is_master():
                    print(f"skipping checkpoint from {latest}: {reason}")
                state = {"step": 0}
            else:
                model.load_state_dict(state["model"], strict=False)
            if "optim" in state:
                loaded_optim, reason = xla_utils.load_optimizer_state_if_compatible(
                    optim, state["optim"]
                )
                if not loaded_optim:
                    if xla_utils.is_master():
                        print(f"skipping optimizer state from {latest}: {reason}")
                    state["step"] = 0
            start_step = int(state.get("step", 0))

    if xla_utils.is_master():
        wandb_utils.init(
            cfg.wandb,
            hyperparams={
                "model": cfg.model.__dict__,
                "lr": cfg.lr,
                "batch_size": cfg.batch_size,
                "total_steps": cfg.total_steps,
                "heads": [s.__dict__ for s in cfg.head_specs],
                "gaussian_loss": cfg.gaussian_loss.__dict__,
            },
            is_master=True,
        )

    step = start_step
    optim.zero_grad(set_to_none=True)
    for batch in device_loader:
        # encode once so we can apply gaussian loss to the layer-idx hidden state
        hidden = model.encode(batch["input_ids"], batch["attention_mask"])
        preds = {name: head(hidden, batch["attention_mask"]) for name, head in model.heads.items()}
        targets = batch["targets"]
        loss, parts = distillation_loss(
            preds,
            targets,
            attention_mask=batch["attention_mask"],
            weights=cfg.teacher_weights,
        )
        if gaussian_fn is not None:
            mask = batch["attention_mask"].bool()
            valid = hidden[mask]  # (sum_T, H) — pool valid frames into the (N, D) input shape
            if valid.shape[0] >= 2:
                gloss = gaussian_fn(valid).total
                loss = loss + cfg.gaussian_loss.weight * gloss
                parts["gaussian"] = float(gloss.detach())

        (loss / cfg.grad_accum).backward()
        if stop.requested:
            optim.zero_grad(set_to_none=True)
            if cfg.ckpt_dir:
                state = {"model": model.state_dict(), "optim": optim.state_dict(), "step": step}
                xla_utils.save_checkpoint(state, f"{cfg.ckpt_dir}/step-{step:08d}.pt")
                xla_utils.save_checkpoint(state, f"{cfg.ckpt_dir}/latest.pt")
            if xla_utils.is_master():
                print(f"stop signal received (signum={stop.signum}); exiting")
            wandb_utils.finish()
            raise SystemExit(130)

        if (step + 1) % cfg.grad_accum == 0:
            for g in optim.param_groups:
                g["lr"] = _lr_at(step // cfg.grad_accum, cfg)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            xla_utils.optimizer_step(optim)
            optim.zero_grad(set_to_none=True)

        if step % cfg.log_every == 0 and xla_utils.is_master():
            parts_s = " ".join(f"{k}={v:.4f}" for k, v in parts.items())
            print(f"step {step:>6d}  total {float(loss):.4f}  {parts_s}")
            metrics = {f"train/{k}": v for k, v in parts.items()}
            metrics["train/total"] = float(loss)
            metrics["train/lr"] = optim.param_groups[0]["lr"]
            wandb_utils.log(metrics, step=step)
        ckpt_due = (
            cfg.ckpt_dir and cfg.ckpt_every > 0 and step > 0 and step % cfg.ckpt_every == 0
        )
        if ckpt_due or (stop.requested and cfg.ckpt_dir):
            state = {"model": model.state_dict(), "optim": optim.state_dict(), "step": step}
            xla_utils.save_checkpoint(state, f"{cfg.ckpt_dir}/step-{step:08d}.pt")
            xla_utils.save_checkpoint(state, f"{cfg.ckpt_dir}/latest.pt")
        if stop.requested:
            if xla_utils.is_master():
                print(f"stop signal received (signum={stop.signum}); exiting")
            wandb_utils.finish()
            raise SystemExit(130)
        step += 1
        if step >= cfg.total_steps:
            break

    if cfg.ckpt_dir:
        state = {"model": model.state_dict(), "optim": optim.state_dict(), "step": step}
        xla_utils.save_checkpoint(state, f"{cfg.ckpt_dir}/final.pt")
        xla_utils.save_checkpoint(state, f"{cfg.ckpt_dir}/latest.pt")
    wandb_utils.finish()


def load_config(path: str | Path) -> FinetuneConfig:
    raw = yaml.safe_load(Path(path).read_text())
    model = SdmConfig(**raw["model"])
    data = NeucodecConfig(**raw["data"])
    head_specs = [HeadSpec(**h) for h in raw["heads"]]
    train_kwargs = dict(raw["train"])
    if "gaussian_loss" in train_kwargs:
        train_kwargs["gaussian_loss"] = GaussianLossConfig(**train_kwargs["gaussian_loss"])
    if "wandb" in train_kwargs:
        train_kwargs["wandb"] = wandb_utils.WandbConfig(**train_kwargs["wandb"])
    return FinetuneConfig(model=model, data=data, head_specs=head_specs, **train_kwargs)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def _main_worker(_index: int, args: argparse.Namespace) -> None:
    train(load_config(args.config))


def main() -> None:
    load_dotenv()
    args = _parse_args()
    xla_utils.launch(_main_worker, args=(args,))


if __name__ == "__main__":
    main()
