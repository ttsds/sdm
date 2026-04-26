"""Single-host MLM pretraining loop. Used both as a smoke test and as the
inner loop the TPU launcher wraps with PyTorch/XLA SPMD sharding.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, IterableDataset

from sdm.data.neucodec_dataset import (
    NeucodecConfig,
    codes_to_input_ids,
    collate,
    iter_examples,
)
from sdm.losses.mlm import MLMConfig, mask_tokens
from sdm.losses.wristband_gaussian import GaussianLossConfig, WristbandGaussianLoss
from sdm.modeling.deberta_neucodec import (
    SDM_BASE,
    SDM_SMALL,
    SdmConfig,
    build_model,
)
from sdm.train import io as ckpt_io
from sdm.train import preempt, wandb_utils, xla_utils


@dataclass
class TrainConfig:
    model: SdmConfig
    data: NeucodecConfig
    mlm: MLMConfig
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    total_steps: int = 100_000
    log_every: int = 50
    grad_accum: int = 1
    seed: int = 0
    ckpt_dir: str | None = None
    ckpt_every: int = 5000
    fsdp: bool = True
    resume_from_latest: bool = True
    gaussian_loss: GaussianLossConfig = field(default_factory=GaussianLossConfig)
    wandb: wandb_utils.WandbConfig = field(default_factory=wandb_utils.WandbConfig)


class _StreamingDataset(IterableDataset):
    def __init__(self, cfg: NeucodecConfig):
        self.cfg = cfg

    def __iter__(self) -> Iterator[dict]:
        return iter_examples(self.cfg)


class _SyntheticDataset(IterableDataset):
    """Generates random FSQ-shaped batches; for offline smoke tests."""

    def __init__(self, cfg: SdmConfig, max_length: int, num_examples: int = 1024):
        self.cfg = cfg
        self.max_length = max_length
        self.num_examples = num_examples

    def __iter__(self) -> Iterator[dict]:
        rng = torch.Generator().manual_seed(0)
        upper = max(8, self.max_length - 2)
        lower = max(4, min(64, upper - 1))
        for i in range(self.num_examples):
            length = int(torch.randint(lower, upper, (1,), generator=rng))
            codes = torch.randint(0, self.cfg.fsq_vocab_size, (length,), generator=rng).tolist()
            input_ids, attn = codes_to_input_ids(codes, self.max_length)
            yield {"input_ids": input_ids, "attention_mask": attn, "id": f"syn-{i}"}


def _lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _save(model, optim, step: int, ckpt_dir: str, *, also_latest: bool = True) -> None:
    state = {"model": model.state_dict(), "optim": optim.state_dict(), "step": step}
    xla_utils.save_checkpoint(state, f"{ckpt_dir}/step-{step:08d}.pt")
    if also_latest:
        xla_utils.save_checkpoint(state, f"{ckpt_dir}/latest.pt")


def train(cfg: TrainConfig, *, synthetic: bool = False) -> None:
    torch.manual_seed(cfg.seed)
    device = xla_utils.get_device(
        require_xla=xla_utils.xla_required() or (cfg.fsdp and not synthetic)
    )
    stop = preempt.install()

    model = build_model(cfg.model).to(device)
    if cfg.fsdp:
        model = xla_utils.shard_module_fsdp(model)
    model.train()

    if synthetic:
        ds: IterableDataset = _SyntheticDataset(cfg.model, cfg.data.max_length)
    else:
        ds = _StreamingDataset(cfg.data)
    loader = DataLoader(ds, batch_size=cfg.batch_size, collate_fn=collate)
    device_loader = xla_utils.loader_per_device(loader, device)

    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optim = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.98), eps=1e-6)

    gaussian_fn: WristbandGaussianLoss | None = None
    if cfg.gaussian_loss.enabled:
        cal_shape = (
            (cfg.batch_size * cfg.data.max_length, cfg.model.hidden_size)
            if cfg.gaussian_loss.calibrate
            else None
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
                "gaussian_loss": cfg.gaussian_loss.__dict__,
            },
            is_master=True,
        )

    step = start_step
    optim.zero_grad(set_to_none=True)
    for batch in device_loader:
        input_ids = batch["input_ids"]
        attn = batch["attention_mask"]
        masked, labels = mask_tokens(input_ids, attn, cfg.model.vocab_size, cfg.mlm)
        out = model(
            input_ids=masked,
            attention_mask=attn,
            labels=labels,
            output_hidden_states=gaussian_fn is not None,
        )
        loss = out.loss / cfg.grad_accum
        gloss_val: float | None = None
        if gaussian_fn is not None:
            hidden = out.hidden_states[-1]
            valid = hidden[attn.bool()]
            if valid.shape[0] >= 2:
                gloss = gaussian_fn(valid).total
                loss = loss + cfg.gaussian_loss.weight * gloss / cfg.grad_accum
                gloss_val = float(gloss.detach())
        loss.backward()

        if (step + 1) % cfg.grad_accum == 0:
            for g in optim.param_groups:
                g["lr"] = _lr_at(step // cfg.grad_accum, cfg)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            xla_utils.optimizer_step(optim)
            optim.zero_grad(set_to_none=True)

        if step % cfg.log_every == 0 and xla_utils.is_master():
            line = (
                f"step {step:>6d}  loss {out.loss.item():.4f}  "
                f"lr {optim.param_groups[0]['lr']:.2e}"
            )
            if gloss_val is not None:
                line += f"  gaussian {gloss_val:.4f}"
            print(line)
            metrics = {"train/loss": float(out.loss.item()), "train/lr": optim.param_groups[0]["lr"]}
            if gloss_val is not None:
                metrics["train/gaussian"] = gloss_val
            wandb_utils.log(metrics, step=step)

        ckpt_due = (
            cfg.ckpt_dir
            and cfg.ckpt_every > 0
            and step > 0
            and step % cfg.ckpt_every == 0
        )
        if ckpt_due or (stop.requested and cfg.ckpt_dir):
            _save(model, optim, step, cfg.ckpt_dir)
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


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=("small", "base"), default="small")
    p.add_argument("--synthetic", action="store_true", help="Use random tokens instead of HF dataset")
    p.add_argument("--total-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--fsq-vocab-size", type=int, default=65536)
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--fsdp", action="store_true", help="Wrap model in FSDP-via-XLA on TPU")
    return p.parse_args()


def main() -> None:
    args = _parse()
    xla_utils.launch(_main_worker, args=(args,))


def _main_worker(_index: int, args: argparse.Namespace) -> None:
    model_cfg = SDM_SMALL if args.size == "small" else SDM_BASE
    model_cfg = SdmConfig(
        fsq_vocab_size=args.fsq_vocab_size,
        hidden_size=model_cfg.hidden_size,
        num_hidden_layers=model_cfg.num_hidden_layers,
        num_attention_heads=model_cfg.num_attention_heads,
        intermediate_size=model_cfg.intermediate_size,
        max_position_embeddings=max(model_cfg.max_position_embeddings, args.max_length),
    )
    cfg = TrainConfig(
        model=model_cfg,
        data=NeucodecConfig(max_length=args.max_length),
        mlm=MLMConfig(),
        batch_size=args.batch_size,
        total_steps=args.total_steps,
        warmup_steps=min(50, args.total_steps // 4),
        ckpt_dir=args.ckpt_dir,
        fsdp=args.fsdp,
    )
    train(cfg, synthetic=args.synthetic)


if __name__ == "__main__":
    main()
