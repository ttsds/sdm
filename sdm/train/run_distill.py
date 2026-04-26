"""YAML entrypoint for the post-pivot mHuBERT distillation runs.

Loads a per-experiment config (configs/finetune_<exp>.yaml), wires up the
streaming Emilia loader, the configured teacher, and the mHuBERT backbone,
then runs the chunk-level MSE distillation loop. Mirrors the
FSDP-via-XLA + cosine-LR + ckpt + wandb shape of the legacy pretrain loop.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from sdm.dotenv import load_dotenv
from sdm.modeling.distill_model import BackboneConfig, build_backbone
from sdm.train import io as ckpt_io
from sdm.train import preempt, wandb_utils, xla_utils


@dataclass
class TeacherConfig:
    kind: str
    target_dim: int
    pooled: str
    in_loop: bool = True
    model_id: str | None = None
    layer: int | None = None
    cache_dir: str | None = None


@dataclass
class EmiliaConfig:
    repo_id: str
    split: str = "train"
    streaming: bool = True
    shuffle_buffer: int = 10000
    seed: int = 0
    fraction: float = 1.0
    sample_rate: int = 16000
    chunk_seconds: float = 1.0
    max_chunks: int = 32
    num_workers: int = 0


@dataclass
class DistillTrainConfig:
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    total_steps: int = 30000
    log_every: int = 50
    ckpt_every: int = 5000
    ckpt_dir: str | None = None
    resume_from_latest: bool = True
    fsdp: bool = True
    seed: int = 0
    wandb: wandb_utils.WandbConfig = field(default_factory=wandb_utils.WandbConfig)


@dataclass
class DistillConfig:
    experiment: str
    backbone: BackboneConfig
    teacher: TeacherConfig
    data: EmiliaConfig
    train: DistillTrainConfig


def load_config(path: str | Path) -> DistillConfig:
    raw = yaml.safe_load(Path(path).read_text())
    train = dict(raw["train"])
    if "wandb" in train:
        train["wandb"] = wandb_utils.WandbConfig(**train["wandb"])
    return DistillConfig(
        experiment=raw["experiment"],
        backbone=BackboneConfig(**raw["backbone"]),
        teacher=TeacherConfig(**raw["teacher"]),
        data=EmiliaConfig(**raw["data"]),
        train=DistillTrainConfig(**train),
    )


def _to_streaming_emilia_cfg(cfg: EmiliaConfig):
    from sdm.data.streaming_emilia import EmiliaConfig as StreamCfg

    return StreamCfg(**cfg.__dict__)


def _to_hf_ssl_cfg(cfg: TeacherConfig):
    from sdm.data.teachers.hf_ssl import HfSslConfig

    if cfg.model_id is None or cfg.layer is None:
        raise ValueError("hf_ssl teacher requires model_id and layer")
    return HfSslConfig(
        model_id=cfg.model_id,
        layer=int(cfg.layer),
        target_dim=int(cfg.target_dim),
        pooled=cfg.pooled,
    )


def _build_loader(cfg: DistillConfig) -> DataLoader:
    from sdm.data.streaming_emilia import StreamingEmiliaDataset, collate

    stream_cfg = _to_streaming_emilia_cfg(cfg.data)
    ds = StreamingEmiliaDataset(stream_cfg)
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        collate_fn=collate,
        num_workers=cfg.data.num_workers,
    )


def _build_teacher(cfg: DistillConfig, device: torch.device) -> nn.Module:
    from sdm.data.teachers import build_teacher

    if cfg.teacher.kind == "hf_ssl":
        from sdm.data.teachers.hf_ssl import HfSslTeacher

        return HfSslTeacher(_to_hf_ssl_cfg(cfg.teacher), device=device)
    return build_teacher(cfg.teacher, device=device)


def _lr_at(step: int, cfg: DistillTrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _save(model, optim, step: int, ckpt_dir: str, *, also_latest: bool = True) -> None:
    state = {"model": model.state_dict(), "optim": optim.state_dict(), "step": step}
    xla_utils.save_checkpoint(state, f"{ckpt_dir}/step-{step:08d}.pt")
    if also_latest:
        xla_utils.save_checkpoint(state, f"{ckpt_dir}/latest.pt")


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, chunk_mask: torch.Tensor) -> torch.Tensor:
    # pred, target: (B, N, D); chunk_mask: (B, N) bool
    mask = chunk_mask.to(pred.dtype).unsqueeze(-1)
    diff = (pred - target) * mask
    denom = mask.sum() * pred.shape[-1]
    return diff.pow(2).sum() / torch.clamp(denom, min=1.0)


def train(cfg: DistillConfig, *, verbose: bool = False) -> None:
    torch.manual_seed(cfg.train.seed)
    device = xla_utils.get_device()
    stop = preempt.install()

    if verbose and xla_utils.is_master():
        print(f"[verbose] experiment={cfg.experiment}")
        print(f"[verbose] device={device}")
        print(f"[verbose] world_size={xla_utils.world_size()}")
        print(f"[verbose] backbone={cfg.backbone.__dict__}")
        print(f"[verbose] teacher={cfg.teacher.__dict__}")
        print(f"[verbose] data={cfg.data.__dict__}")
        print(f"[verbose] train={cfg.train.__dict__}")

    t0 = time.perf_counter()
    student = build_backbone(cfg.backbone, target_dim=cfg.teacher.target_dim).to(device)
    if cfg.train.fsdp:
        student = xla_utils.shard_module_fsdp(student)
    student.train()
    if verbose and xla_utils.is_master():
        n_params = sum(p.numel() for p in student.parameters())
        print(f"[verbose] student built ({n_params/1e6:.1f}M params) in {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    teacher = _build_teacher(cfg, device)
    teacher.eval()
    if verbose and xla_utils.is_master():
        print(f"[verbose] teacher built in {time.perf_counter()-t0:.2f}s")

    loader = _build_loader(cfg)
    device_loader = xla_utils.loader_per_device(loader, device)

    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {
            "params": [p for n, p in student.named_parameters() if not any(k in n for k in no_decay)],
            "weight_decay": cfg.train.weight_decay,
        },
        {
            "params": [p for n, p in student.named_parameters() if any(k in n for k in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optim = torch.optim.AdamW(params, lr=cfg.train.lr, betas=(0.9, 0.98), eps=1e-6)

    start_step = 0
    if cfg.train.ckpt_dir and cfg.train.resume_from_latest:
        latest = ckpt_io.latest_checkpoint(cfg.train.ckpt_dir)
        if latest is not None:
            if xla_utils.is_master():
                print(f"resuming from {latest}")
            state = ckpt_io.load_state(latest, map_location="cpu")
            student.load_state_dict(state["model"], strict=False)
            if "optim" in state:
                optim.load_state_dict(state["optim"])
            start_step = int(state.get("step", 0))

    if xla_utils.is_master():
        wandb_utils.init(
            cfg.train.wandb,
            hyperparams={
                "experiment": cfg.experiment,
                "backbone": cfg.backbone.__dict__,
                "teacher": cfg.teacher.__dict__,
                "data": cfg.data.__dict__,
                "lr": cfg.train.lr,
                "batch_size": cfg.train.batch_size,
                "total_steps": cfg.train.total_steps,
            },
            is_master=True,
        )

    step = start_step
    optim.zero_grad(set_to_none=True)
    if verbose and xla_utils.is_master():
        xla_utils.clear_metrics()
        print("[verbose] entering training loop; waiting for first batch ...")
    _t_prev = time.perf_counter()
    for batch in device_loader:
        if verbose and xla_utils.is_master():
            t_data = time.perf_counter() - _t_prev
        audio = batch["audio"]
        chunk_mask = batch["chunk_mask"]

        if verbose and xla_utils.is_master():
            t0 = time.perf_counter()
        with torch.no_grad():
            target = teacher(audio, chunk_mask=chunk_mask)
        if verbose and xla_utils.is_master():
            xla_utils.mark_step()  # force materialization for accurate timing
            t_teacher = time.perf_counter() - t0
            t0 = time.perf_counter()

        pred = student(audio)
        loss = _masked_mse(pred, target, chunk_mask) / cfg.train.grad_accum
        loss.backward()
        if verbose and xla_utils.is_master():
            xla_utils.mark_step()
            t_student = time.perf_counter() - t0
            t0 = time.perf_counter()

        if (step + 1) % cfg.train.grad_accum == 0:
            for g in optim.param_groups:
                g["lr"] = _lr_at(step // cfg.train.grad_accum, cfg.train)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            xla_utils.reduce_gradients(optim)
            optim.step()
            optim.zero_grad(set_to_none=True)
            xla_utils.mark_step()
        if verbose and xla_utils.is_master():
            t_optim = time.perf_counter() - t0
            metrics = xla_utils.compile_metrics()
            compile_msg = ""
            if metrics:
                compile_msg = (
                    "  xla_compile "
                    f"samples={metrics.get('CompileTimeSamples', 0)} "
                    f"uncached={metrics.get('UncachedCompile', 0)} "
                    f"cached={metrics.get('CachedCompile', 0)} "
                    f"handles={metrics.get('CreateCompileHandles', 0)}"
                )
            print(
                f"[verbose] step {step:>6d}  "
                f"data {t_data*1e3:7.1f}ms  "
                f"teacher {t_teacher*1e3:7.1f}ms  "
                f"student {t_student*1e3:7.1f}ms  "
                f"optim {t_optim*1e3:7.1f}ms  "
                f"audio {tuple(audio.shape)}"
                f"{compile_msg}"
            )

        if step % cfg.train.log_every == 0 and xla_utils.is_master():
            loss_val = float(loss.detach()) * cfg.train.grad_accum
            print(
                f"step {step:>6d}  loss {loss_val:.4f}  lr {optim.param_groups[0]['lr']:.2e}"
            )
            wandb_utils.log(
                {"train/loss": loss_val, "train/lr": optim.param_groups[0]["lr"]},
                step=step,
            )

        ckpt_due = (
            cfg.train.ckpt_dir
            and cfg.train.ckpt_every > 0
            and step > 0
            and step % cfg.train.ckpt_every == 0
        )
        if ckpt_due or (stop.requested and cfg.train.ckpt_dir):
            _save(student, optim, step, cfg.train.ckpt_dir)
        if stop.requested:
            if xla_utils.is_master():
                print(f"stop signal received (signum={stop.signum}); exiting")
            wandb_utils.finish()
            raise SystemExit(130)
        step += 1
        if step >= cfg.train.total_steps:
            break
        if verbose and xla_utils.is_master():
            _t_prev = time.perf_counter()

    if cfg.train.ckpt_dir:
        state = {"model": student.state_dict(), "optim": optim.state_dict(), "step": step}
        xla_utils.save_checkpoint(state, f"{cfg.train.ckpt_dir}/final.pt")
        xla_utils.save_checkpoint(state, f"{cfg.train.ckpt_dir}/latest.pt")
    wandb_utils.finish()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Load config and check imports only")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-step phase timings (data/teacher/student/optim) and startup info",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dry_run:
        import sdm.data.streaming_emilia  # noqa: F401
        import sdm.data.teachers  # noqa: F401

        print(f"dry-run OK: experiment={cfg.experiment} teacher.kind={cfg.teacher.kind}")
        return

    train(cfg, verbose=args.verbose)


if __name__ == "__main__":
    main()
