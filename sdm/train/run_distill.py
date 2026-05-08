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
from sdm.losses.wristband_gaussian import GaussianLossConfig, WristbandGaussianLoss
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
    target_layernorm: bool = True
    # Distillation loss: "cos_l1" (default, dense embeddings),
    # "cos" (direction-only, e.g. speaker embeddings),
    # "l1" (scalar regression), or "mse".
    loss: str = "cos_l1"
    # Optional, teacher-specific extras (e.g. g2p speaking-rate, pyworld_f0).
    chunk_seconds: float | None = None
    sample_rate: int | None = None
    default_language: str | None = None
    count_mode: str | None = None
    frame_period_ms: float | None = None
    f0_floor: float | None = None
    f0_ceil: float | None = None
    teacher_sample_rate: int | None = None
    # FunASR AutoModel hub: "hf" (default) or "ms" -- emotion2vec teacher.
    hub: str | None = None
    # Linear target normalisation applied inside scalar teachers
    # (pyworld_f0, g2p_speaking_rate). out -> (out - mean) / scale.
    target_mean: float | None = None
    target_scale: float | None = None
    # Allosaurus speaking-rate extras (audio-based per-chunk teacher).
    allosaurus_model: str | None = None
    emit: float | None = None


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
    # Held-out eval cadence in optimizer steps. <=0 disables in-loop eval.
    eval_every: int = 0
    # Number of held-out utterances to draw at startup (separate stream seed).
    eval_utterances: int = 64
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
    # Optional Gaussian-wristband regularizer applied to the pre-head
    # pooled encoder output. Disabled by default (no-op for old runs).
    regularizer: GaussianLossConfig = field(default_factory=GaussianLossConfig)


def load_config(path: str | Path) -> DistillConfig:
    raw = yaml.safe_load(Path(path).read_text())
    train = dict(raw["train"])
    if "wandb" in train:
        train["wandb"] = wandb_utils.WandbConfig(**train["wandb"])
    reg_block = raw.get("regularizer") or {}
    return DistillConfig(
        experiment=raw["experiment"],
        backbone=BackboneConfig(**raw["backbone"]),
        teacher=TeacherConfig(**raw["teacher"]),
        data=EmiliaConfig(**raw["data"]),
        train=DistillTrainConfig(**train),
        regularizer=GaussianLossConfig(**reg_block),
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
        target_layernorm=bool(cfg.target_layernorm),
    )


def _build_loader(cfg: DistillConfig) -> DataLoader:
    from sdm.data.streaming_emilia import StreamingEmiliaDataset, make_collate

    stream_cfg = _to_streaming_emilia_cfg(cfg.data)
    ds = StreamingEmiliaDataset(stream_cfg)
    teacher_id = (
        cfg.teacher.model_id
        if cfg.teacher.kind in {"hf_ssl", "hf_ctc"}
        else None
    )
    student_id = (
        cfg.backbone.model_id
        if cfg.backbone.kind in {"hf", "fairseq_w2v2"}
        else None
    )
    collate_fn = make_collate(
        teacher_processor_id=teacher_id,
        student_processor_id=student_id,
        sample_rate=cfg.data.sample_rate,
    )
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        collate_fn=collate_fn,
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
    pred = pred.float()
    target = target.float()
    mask = chunk_mask.to(pred.dtype).unsqueeze(-1)
    diff = (pred - target) * mask
    denom = mask.sum() * pred.shape[-1]
    return diff.pow(2).sum() / torch.clamp(denom, min=1.0)


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, chunk_mask: torch.Tensor) -> torch.Tensor:
    """Plain masked L1, averaged over masked chunk positions.

    Used for scalar regression targets (``target_dim == 1``) where the
    cosine term in :func:`_masked_cos_l1` collapses to ``sign(pred) *
    sign(target)`` and contributes no useful gradient w.r.t. magnitude.
    """
    pred_f = pred.float()
    target_f = target.float()
    mask = chunk_mask.to(pred_f.dtype)
    l1_per_token = (pred_f - target_f).abs().mean(dim=-1) * mask
    denom = torch.clamp(mask.sum(), min=1.0)
    return l1_per_token.sum() / denom


def _masked_cos(
    pred: torch.Tensor, target: torch.Tensor, chunk_mask: torch.Tensor
) -> torch.Tensor:
    """Direction-only cosine distillation loss.

    For every masked token position computes ``1 - cos(pred, target)`` and
    averages over masked positions. Magnitude carries no information for
    speaker-embedding teachers (dvector, wespeaker) where only direction
    on the unit hypersphere encodes speaker identity, so dropping the L1
    term avoids penalising the student for arbitrary scale differences.
    """
    pred_f = pred.float()
    target_f = target.float()
    mask = chunk_mask.to(pred_f.dtype)
    cos = torch.nn.functional.cosine_similarity(pred_f, target_f, dim=-1, eps=1e-6)
    cos_per_token = (1.0 - cos) * mask
    denom = torch.clamp(mask.sum(), min=1.0)
    return cos_per_token.sum() / denom


def _masked_cos_l1(
    pred: torch.Tensor, target: torch.Tensor, chunk_mask: torch.Tensor
) -> torch.Tensor:
    """Lean cosine + L1 distillation loss used in the hot training path.

    For every masked token position computes
    ``0.5 * (1 - cos(pred, target)) + 0.5 * mean_d |pred - target|`` and
    averages over masked positions. The cosine term is direction-only; the
    L1 term keeps magnitudes anchored to the teacher.
    """
    pred_f = pred.float()
    target_f = target.float()
    mask = chunk_mask.to(pred_f.dtype)
    cos = torch.nn.functional.cosine_similarity(pred_f, target_f, dim=-1, eps=1e-6)
    cos_per_token = (1.0 - cos) * mask
    l1_per_token = (pred_f - target_f).abs().mean(dim=-1) * mask
    denom = torch.clamp(mask.sum(), min=1.0)
    cos_loss = cos_per_token.sum() / denom
    l1_loss = l1_per_token.sum() / denom
    return 0.5 * cos_loss + 0.5 * l1_loss


def _masked_cos_l1_terms(
    pred: torch.Tensor, target: torch.Tensor, chunk_mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Diagnostic version of :func:`_masked_cos_l1` for ``--verbose`` runs.

    Returns each intermediate so the caller can print stats. The returned
    ``loss`` matches what the hot path produces.
    """
    pred_f = pred.float()
    target_f = target.float()
    mask = chunk_mask.to(pred_f.dtype)
    cos = torch.nn.functional.cosine_similarity(pred_f, target_f, dim=-1, eps=1e-6)
    cos_per_token = (1.0 - cos) * mask
    abs_diff = (pred_f - target_f).abs()
    l1_per_token = abs_diff.mean(dim=-1) * mask
    mask_sum = mask.sum()
    denom = torch.clamp(mask_sum, min=1.0)
    cos_loss = cos_per_token.sum() / denom
    l1_loss = l1_per_token.sum() / denom
    loss = 0.5 * cos_loss + 0.5 * l1_loss
    return {
        "pred": pred_f,
        "target": target_f,
        "abs_diff": abs_diff,
        "cos": cos,
        "cos_per_token": cos_per_token,
        "l1_per_token": l1_per_token,
        "cos_loss": cos_loss,
        "l1_loss": l1_loss,
        "mask_sum": mask_sum,
        "loss": loss,
    }


def _stats_for_debug(name: str, tensor: torch.Tensor) -> str:
    value = tensor.detach().float()
    finite = torch.isfinite(value)
    finite_count = int(finite.sum().cpu().item())
    total = value.numel()
    clean = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    return (
        f"{name}: shape={tuple(value.shape)} finite={finite_count}/{total} "
        f"min={float(clean.min().cpu()):.4g} max={float(clean.max().cpu()):.4g} "
        f"mean={float(clean.mean().cpu()):.4g}"
    )


def _distill_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    chunk_mask: torch.Tensor,
    loss_kind: str,
) -> torch.Tensor:
    """Dispatch the per-teacher distillation loss without grad_accum scaling.

    Centralised so the held-out eval path uses exactly the same arithmetic
    as the training step (just without the 1/grad_accum divide).
    """
    if loss_kind == "cos_l1":
        return _masked_cos_l1(pred, target, chunk_mask)
    if loss_kind == "cos":
        return _masked_cos(pred, target, chunk_mask)
    if loss_kind == "l1":
        return _masked_l1(pred, target, chunk_mask)
    if loss_kind == "mse":
        return _masked_mse(pred, target, chunk_mask)
    raise ValueError(f"unknown teacher.loss={loss_kind!r}")


def _wristband_term(
    encoded: torch.Tensor,
    chunk_mask: torch.Tensor,
    loss_fn: WristbandGaussianLoss,
):
    """Apply the wristband loss to flattened valid pre-head latents.

    ``encoded`` is ``(B, N, D)``; we gather rows where ``chunk_mask`` is
    true and pass them as ``(1, n_valid, D)`` so the loss treats the whole
    batch's valid chunks as a single Gaussian sample. Returns ``None``
    when fewer than 2 valid rows are available (the loss requires N>=2
    for pairwise/covariance terms).
    """
    mask = chunk_mask.to(torch.bool)
    if int(mask.sum()) < 2:
        return None
    latents = encoded[mask].to(torch.float32).unsqueeze(0)
    return loss_fn(latents)


def _build_held_out_batches(cfg: "DistillConfig") -> list[dict] | None:
    """Pull a fixed held-out batch list from a separate-seed Emilia stream.

    Run once at startup on the master process. We deliberately use a
    different stream seed (``cfg.data.seed + 100003``) so the held-out
    utterances do not collide with the training stream's prefix; the
    Emilia corpus is large enough that this is effectively a disjoint
    sample at the 10% fraction we use.

    Tensors are kept on CPU; the eval loop moves each batch to ``device``
    on demand to avoid pinning GPU memory across the run.
    """
    if cfg.train.eval_every <= 0 or cfg.train.eval_utterances <= 0:
        return None
    from sdm.data.streaming_emilia import StreamingEmiliaDataset, make_collate

    stream_cfg = _to_streaming_emilia_cfg(cfg.data)
    stream_cfg.seed = int(cfg.data.seed) + 100003
    ds = StreamingEmiliaDataset(stream_cfg)
    teacher_id = (
        cfg.teacher.model_id
        if cfg.teacher.kind in {"hf_ssl", "hf_ctc"}
        else None
    )
    student_id = (
        cfg.backbone.model_id
        if cfg.backbone.kind in {"hf", "fairseq_w2v2"}
        else None
    )
    collate_fn = make_collate(
        teacher_processor_id=teacher_id,
        student_processor_id=student_id,
        sample_rate=cfg.data.sample_rate,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        collate_fn=collate_fn,
        num_workers=0,
    )
    n_target = int(cfg.train.eval_utterances)
    seen = 0
    batches: list[dict] = []
    for batch in loader:
        cpu_batch = {
            k: (v.detach().cpu() if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        batches.append(cpu_batch)
        seen += int(cpu_batch["audio"].shape[0])
        if seen >= n_target:
            break
    return batches or None


@torch.no_grad()
def _run_held_out_eval(
    student: nn.Module,
    teacher: nn.Module,
    batches: list[dict],
    *,
    loss_kind: str,
    device: torch.device,
    wristband: WristbandGaussianLoss | None,
) -> dict[str, float]:
    """Run distill loss (and optional wristband) on the held-out batches.

    Returns a dict ready for ``wandb_utils.log``. ``eval/loss`` is the
    bare distillation loss -- the same metric the pre-wristband runs
    logged as ``train/loss`` -- so wristband-on / wristband-off runs can
    be compared directly without re-running the old experiments.
    """
    student_was_training = student.training
    student.eval()
    distill_sum = 0.0
    distill_w = 0.0
    rep_total_sum = 0.0
    rep_rep_sum = 0.0
    rep_rad_sum = 0.0
    rep_mom_sum = 0.0
    rep_w = 0.0
    for batch in batches:
        chunk_mask = batch["chunk_mask"].to(device, non_blocking=True)
        teacher_audio = batch.get("teacher_audio", batch["audio"]).to(
            device, non_blocking=True
        )
        student_audio = batch.get("student_audio", batch["audio"]).to(
            device, non_blocking=True
        )
        teacher_ctx = {
            "texts": batch.get("texts"),
            "languages": batch.get("languages"),
            "n_chunks": batch.get("n_chunks"),
        }
        target = teacher(teacher_audio, chunk_mask=chunk_mask, **teacher_ctx)
        if wristband is not None:
            pred, encoded = student(student_audio, return_encoded=True)
        else:
            pred = student(student_audio)
            encoded = None
        loss = _distill_loss(pred, target, chunk_mask, loss_kind)
        n_valid = float(chunk_mask.sum().detach().cpu())
        distill_sum += float(loss.detach().cpu()) * n_valid
        distill_w += n_valid
        if wristband is not None and encoded is not None:
            comp = _wristband_term(encoded, chunk_mask, wristband)
            if comp is not None:
                rep_total_sum += float(comp.total.detach().cpu()) * n_valid
                rep_rep_sum += float(comp.rep.detach().cpu()) * n_valid
                rep_rad_sum += float(comp.rad.detach().cpu()) * n_valid
                rep_mom_sum += float(comp.mom.detach().cpu()) * n_valid
                rep_w += n_valid
    if student_was_training:
        student.train()

    if distill_w == 0:
        return {}
    out = {
        "eval/loss": distill_sum / distill_w,
        "eval/n_valid_chunks": distill_w,
    }
    if rep_w > 0:
        out["eval/wristband/total"] = rep_total_sum / rep_w
        out["eval/wristband/rep"] = rep_rep_sum / rep_w
        out["eval/wristband/rad"] = rep_rad_sum / rep_w
        out["eval/wristband/mom"] = rep_mom_sum / rep_w
    return out


def train(cfg: DistillConfig, *, verbose: bool = False) -> None:
    torch.manual_seed(cfg.train.seed)
    device = xla_utils.get_device(
        require_xla=xla_utils.xla_required() or cfg.train.fsdp
    )
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

    # Wristband regularizer (no-op when ``cfg.regularizer.enabled`` is False).
    # We deliberately skip the loss's built-in calibrator: chunk counts vary
    # per batch (1..32 valid chunks), so a fixed calibration_shape would
    # raise. With calibrate=False the per-component (mean=0, std=1) defaults
    # in ``WristbandGaussianLoss`` reduce to identity normalisation and the
    # raw component values are used directly.
    wristband: WristbandGaussianLoss | None = None
    if cfg.regularizer.enabled:
        wristband = WristbandGaussianLoss(
            beta=cfg.regularizer.beta,
            lambda_rad=cfg.regularizer.lambda_rad,
            lambda_mom=cfg.regularizer.lambda_mom,
            moment=cfg.regularizer.moment,
            calibration_shape=None,
        )
        if xla_utils.is_master():
            print(
                f"[wristband] enabled weight={cfg.regularizer.weight} "
                f"beta={cfg.regularizer.beta} lambda_rad={cfg.regularizer.lambda_rad} "
                f"lambda_mom={cfg.regularizer.lambda_mom} moment={cfg.regularizer.moment!r}"
            )

    held_out_batches: list[dict] | None = None
    if cfg.train.eval_every > 0 and xla_utils.is_master():
        t0 = time.perf_counter()
        held_out_batches = _build_held_out_batches(cfg)
        if held_out_batches is not None:
            n_utts = sum(int(b["audio"].shape[0]) for b in held_out_batches)
            print(
                f"[eval] held-out: {n_utts} utterances in {len(held_out_batches)} batches "
                f"(built in {time.perf_counter()-t0:.1f}s, eval_every={cfg.train.eval_every})"
            )

    start_step = 0
    if cfg.train.ckpt_dir and cfg.train.resume_from_latest:
        latest = ckpt_io.latest_checkpoint(cfg.train.ckpt_dir)
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
                student.load_state_dict(state["model"], strict=False)
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
        teacher_audio = batch.get("teacher_audio", audio)
        student_audio = batch.get("student_audio", audio)
        teacher_ctx = {
            "texts": batch.get("texts"),
            "languages": batch.get("languages"),
            "n_chunks": batch.get("n_chunks"),
        }

        if verbose and xla_utils.is_master():
            t0 = time.perf_counter()
        with torch.no_grad():
            target = teacher(teacher_audio, chunk_mask=chunk_mask, **teacher_ctx)
        if verbose and xla_utils.is_master():
            xla_utils.mark_step()  # force materialization for accurate timing
            t_teacher = time.perf_counter() - t0
            t0 = time.perf_counter()

        # Single backbone forward. When the regularizer is on we ask the
        # student to return both the head output and the pooled encoder
        # output, so FSDP only all-gathers parameters once. (Calling
        # ``student.encode`` separately under FSDP would require a second
        # full forward to re-trace, doubling the compute on the entire
        # backbone.)
        encoded: torch.Tensor | None = None
        if wristband is not None:
            pred, encoded = student(student_audio, return_encoded=True)
        else:
            pred = student(student_audio)
        loss_kind = cfg.teacher.loss
        if loss_kind == "cos_l1":
            if verbose and xla_utils.is_master():
                loss_terms = _masked_cos_l1_terms(pred, target, chunk_mask)
                distill = loss_terms["loss"]
            else:
                loss_terms = None
                distill = _masked_cos_l1(pred, target, chunk_mask)
        elif loss_kind == "cos":
            loss_terms = None
            distill = _masked_cos(pred, target, chunk_mask)
        elif loss_kind == "l1":
            loss_terms = None
            distill = _masked_l1(pred, target, chunk_mask)
        elif loss_kind == "mse":
            loss_terms = None
            distill = _masked_mse(pred, target, chunk_mask)
        else:
            raise ValueError(f"unknown teacher.loss={loss_kind!r}")

        wristband_comp = None
        if wristband is not None and encoded is not None:
            wristband_comp = _wristband_term(encoded, chunk_mask, wristband)

        if wristband_comp is not None:
            total_loss = distill + cfg.regularizer.weight * wristband_comp.total
        else:
            total_loss = distill
        loss = total_loss / cfg.train.grad_accum
        loss.backward()
        if verbose and xla_utils.is_master():
            xla_utils.mark_step()
            t_student = time.perf_counter() - t0
            t0 = time.perf_counter()

        grad_norm: torch.Tensor | None = None
        skipped_step = False
        if (step + 1) % cfg.train.grad_accum == 0:
            for g in optim.param_groups:
                g["lr"] = _lr_at(step // cfg.train.grad_accum, cfg.train)
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                xla_utils.optimizer_step(optim)
            else:
                skipped_step = True
                if xla_utils.is_master():
                    print(
                        f"[warn] step {step}: non-finite grad_norm "
                        f"({float(grad_norm.detach().cpu()):.4g}); skipping optimizer step"
                    )
            optim.zero_grad(set_to_none=True)
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
            # ``train/loss`` mirrors the pre-wristband definition (the bare
            # distillation loss) so wristband-on / wristband-off W&B runs
            # are directly overlay-able. ``train/total_loss`` is the
            # actual quantity backprop'd; identical when the regularizer is
            # disabled.
            distill_val = float(distill.detach())
            total_val = float(total_loss.detach())
            if verbose and loss_terms is not None:
                print(f"[verbose] step {step} tensor diagnostics:")
                print("[verbose] " + _stats_for_debug("audio", audio))
                print("[verbose] " + _stats_for_debug("teacher_audio", teacher_audio))
                print("[verbose] " + _stats_for_debug("chunk_mask", chunk_mask))
                print("[verbose] " + _stats_for_debug("target", loss_terms["target"]))
                print("[verbose] " + _stats_for_debug("pred", loss_terms["pred"]))
                print("[verbose] " + _stats_for_debug("abs_diff", loss_terms["abs_diff"]))
                print("[verbose] " + _stats_for_debug("cos", loss_terms["cos"]))
                print("[verbose] " + _stats_for_debug("cos_per_token", loss_terms["cos_per_token"]))
                print("[verbose] " + _stats_for_debug("l1_per_token", loss_terms["l1_per_token"]))
                print(
                    f"[verbose] mask_sum={float(loss_terms['mask_sum'].detach().cpu()):.4g} "
                    f"cos_loss={float(loss_terms['cos_loss'].detach().cpu()):.4g} "
                    f"l1_loss={float(loss_terms['l1_loss'].detach().cpu()):.4g}"
                )
                print("[verbose] " + _stats_for_debug("loss", loss))
                if grad_norm is not None:
                    gn = float(grad_norm.detach().cpu())
                    print(
                        f"[verbose] grad_norm={gn:.4g} finite={math.isfinite(gn)} "
                        f"skipped_step={skipped_step}"
                    )
            wristband_str = ""
            if wristband_comp is not None:
                wristband_str = (
                    f"  wristband {float(wristband_comp.total.detach()):.4f}"
                    f" total {total_val:.4f}"
                )
            print(
                f"step {step:>6d}  loss {distill_val:.4f}{wristband_str}  "
                f"lr {optim.param_groups[0]['lr']:.2e}"
            )
            log_payload: dict[str, float] = {
                "train/loss": distill_val,
                "train/total_loss": total_val,
                "train/lr": optim.param_groups[0]["lr"],
            }
            if wristband_comp is not None:
                log_payload["train/wristband/total"] = float(wristband_comp.total.detach())
                log_payload["train/wristband/rep"] = float(wristband_comp.rep.detach())
                log_payload["train/wristband/rad"] = float(wristband_comp.rad.detach())
                log_payload["train/wristband/mom"] = float(wristband_comp.mom.detach())
            wandb_utils.log(log_payload, step=step)

        # Held-out eval (master-only). Runs after the optimizer step on the
        # current `step`; reports both the bare distill loss (for direct
        # comparison with pre-wristband baselines) and the wristband
        # components when enabled.
        eval_due = (
            held_out_batches is not None
            and cfg.train.eval_every > 0
            and step > 0
            and step % cfg.train.eval_every == 0
            and xla_utils.is_master()
        )
        if eval_due:
            t0 = time.perf_counter()
            eval_metrics = _run_held_out_eval(
                student,
                teacher,
                held_out_batches,
                loss_kind=loss_kind,
                device=device,
                wristband=wristband,
            )
            if eval_metrics:
                eval_str = " ".join(
                    f"{k.split('/')[-1]}={v:.4f}" for k, v in eval_metrics.items()
                )
                print(
                    f"[eval] step {step:>6d}  {eval_str}  "
                    f"({time.perf_counter()-t0:.1f}s)"
                )
                wandb_utils.log(eval_metrics, step=step)

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Load config and check imports only")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-step phase timings (data/teacher/student/optim) and startup info",
    )
    return parser.parse_args()


def _run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.dry_run:
        import sdm.data.streaming_emilia  # noqa: F401
        import sdm.data.teachers  # noqa: F401

        print(f"dry-run OK: experiment={cfg.experiment} teacher.kind={cfg.teacher.kind}")
        return

    train(cfg, verbose=args.verbose)


def _main_worker(_index: int, args: argparse.Namespace) -> None:
    _run(args)


def main() -> None:
    load_dotenv()
    args = _parse_args()
    if args.dry_run:
        _run(args)
        return
    xla_utils.launch(_main_worker, args=(args,))


if __name__ == "__main__":
    main()
