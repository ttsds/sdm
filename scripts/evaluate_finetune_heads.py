"""Evaluate the trained per-experiment distillation heads on held-out data.

For each finetune experiment, load the raw ``final.pt`` checkpoint (including
the trained projection head), run it directly on the same held-out split used
by the probe study, and compare the head outputs to that experiment's own
teacher targets.

This answers a different question from ``run_linear_probes.py``:

- probes ask whether a backbone *contains* information about a target
- this script asks whether the *trained head itself* predicts its target well

Usage:
    /home/cdminix/Documents/repos/sdm/sdm/.venv/bin/python scripts/evaluate_finetune_heads.py \
        --out analysis/head_eval/20260507T073739Z \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import r2_score

from scripts.run_linear_probes import (
    _fmt_report,
    _nonfinite_report,
    _stack_batch,
    collect_held_out,
    teacher_targets,
)
from sdm.modeling.distill_model import build_backbone
from sdm.train.run_distill import _build_teacher, load_config


DEFAULT_EXPERIMENTS = [
    "sdm-xlsr-fairseq",
    "sdm-dvector",
    "sdm-pitch",
    "sdm-mpm",
    "sdm-speaking-rate",
    "sdm-w2v2-asr",
    "sdm-mwhisper",
    "sdm-emotion2vec",
]


def gcs_pull(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gsutil", "-q", "cp", src, str(dst)], check=True)


def head_outputs(student, held_out: list[dict], *, device: torch.device, batch_size: int) -> np.ndarray:
    out: list[np.ndarray] = []
    student.eval()
    n_batches = (len(held_out) + batch_size - 1) // batch_size
    for start in range(0, len(held_out), batch_size):
        batch = held_out[start : start + batch_size]
        audio, mask, _ = _stack_batch(batch)
        with torch.no_grad():
            pred = student(audio.to(device))
        valid = pred[mask.to(pred.device)]
        out.append(valid.detach().to(torch.float32).cpu().numpy())
    if not out:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(out)


def masked_loss(pred: np.ndarray, target: np.ndarray, *, loss_kind: str) -> float:
    pred_t = torch.from_numpy(pred).to(torch.float32)
    tgt_t = torch.from_numpy(target).to(torch.float32)
    if loss_kind == "cos":
        cos = torch.nn.functional.cosine_similarity(pred_t, tgt_t, dim=-1, eps=1e-6)
        return float((1.0 - cos).mean().item())
    if loss_kind == "l1":
        return float((pred_t - tgt_t).abs().mean().item())
    if loss_kind == "mse":
        return float(torch.mean((pred_t - tgt_t) ** 2).item())
    if loss_kind == "cos_l1":
        cos = torch.nn.functional.cosine_similarity(pred_t, tgt_t, dim=-1, eps=1e-6)
        l1 = (pred_t - tgt_t).abs().mean(dim=-1)
        return float((0.5 * (1.0 - cos) + 0.5 * l1).mean().item())
    raise ValueError(f"unknown loss_kind {loss_kind!r}")


def evaluate_pair(pred: np.ndarray, target: np.ndarray, *, loss_kind: str) -> dict:
    pred_finite = np.isfinite(pred).reshape(pred.shape[0], -1).all(axis=1)
    tgt_finite = np.isfinite(target).reshape(target.shape[0], -1).all(axis=1)
    keep = pred_finite & tgt_finite
    dropped = int((~keep).sum())
    pred = pred[keep]
    target = target[keep]
    pred_arr = pred.reshape(pred.shape[0], -1) if pred.ndim > 1 else pred.reshape(-1, 1)
    tgt_arr = target.reshape(target.shape[0], -1) if target.ndim > 1 else target.reshape(-1, 1)
    target_var = float(tgt_arr.var(axis=0).mean()) if tgt_arr.size else 0.0
    target_constant = bool(target_var < 1e-12)
    r2 = float("nan")
    if pred.shape[0] and not target_constant:
        r2 = float(r2_score(target, pred, multioutput="uniform_average"))
    cosine_mean = float("nan")
    if pred.shape[0] and pred_arr.shape[1] > 1:
        p = torch.from_numpy(pred_arr).to(torch.float32)
        t = torch.from_numpy(tgt_arr).to(torch.float32)
        cosine_mean = float(torch.nn.functional.cosine_similarity(p, t, dim=-1, eps=1e-6).mean().item())
    return {
        "n_eval": int(pred.shape[0]),
        "n_dropped": dropped,
        "target_var": target_var,
        "target_constant": target_constant,
        "loss": masked_loss(pred, target, loss_kind=loss_kind) if pred.shape[0] else float("nan"),
        "r2": r2,
        "mean_abs_err": float(np.abs(pred_arr - tgt_arr).mean()) if pred.shape[0] else float("nan"),
        "cosine_mean": cosine_mean,
    }


def load_student_with_head(cfg_path: Path, ckpt_path: Path, *, device: torch.device):
    cfg = load_config(cfg_path)
    student = build_backbone(cfg.backbone, target_dim=cfg.teacher.target_dim)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    student.load_state_dict(state, strict=False)
    student.eval()
    student.to(device)
    return cfg, student


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    ap.add_argument("--bucket", default="gs://sdm-ckpts")
    ap.add_argument("--dataset", default="mythicinfinity/libritts")
    ap.add_argument("--dataset-config", default="dev")
    ap.add_argument("--split", default="dev.clean")
    ap.add_argument("--probe-utterances", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--checkpoint-cache", type=Path, default=Path("checkpoints/with_heads"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[head-eval] device={device} experiments={args.experiments}")
    held_out = collect_held_out(
        args.dataset,
        args.split,
        args.probe_utterances,
        config_name=args.dataset_config,
    )
    print(
        f"[head-eval] held_out={len(held_out)} utterances "
        f"({sum(int(r['n_chunks']) for r in held_out)} valid chunks)"
    )

    results = []
    for exp in args.experiments:
        cfg_path = Path(f"configs/finetune_{exp.replace('sdm-', '').replace('-fairseq', '_fairseq').replace('-', '_')}.yaml")
        if not cfg_path.exists():
            cfg_path = Path({
                "sdm-xlsr-fairseq": "configs/finetune_xlsr_fairseq.yaml",
                "sdm-dvector": "configs/finetune_dvector.yaml",
                "sdm-pitch": "configs/finetune_pitch.yaml",
                "sdm-mpm": "configs/finetune_mpm.yaml",
                "sdm-speaking-rate": "configs/finetune_speaking_rate.yaml",
                "sdm-w2v2-asr": "configs/finetune_w2v2_asr.yaml",
                "sdm-mwhisper": "configs/finetune_mwhisper.yaml",
                "sdm-emotion2vec": "configs/finetune_emotion2vec.yaml",
            }[exp])
        ckpt_path = args.checkpoint_cache / exp / "final.pt"
        if not ckpt_path.exists():
            src = f"{args.bucket}/{exp}/final.pt"
            print(f"[pull] {src} -> {ckpt_path}")
            gcs_pull(src, ckpt_path)

        t0 = time.perf_counter()
        cfg, student = load_student_with_head(cfg_path, ckpt_path, device=device)
        teacher = _build_teacher(cfg, device)
        teacher.eval()
        t_load = time.perf_counter() - t0

        t0 = time.perf_counter()
        target = teacher_targets(teacher, held_out, device=device, batch_size=args.batch_size, desc=f"teacher[{exp}]")
        pred = head_outputs(student, held_out, device=device, batch_size=args.batch_size)
        t_eval = time.perf_counter() - t0
        pred_rep = _nonfinite_report(pred)
        tgt_rep = _nonfinite_report(target)
        metrics = evaluate_pair(pred, target, loss_kind=cfg.teacher.loss)
        row = {
            "experiment": exp,
            "teacher_kind": cfg.teacher.kind,
            "loss_kind": cfg.teacher.loss,
            "target_dim": int(cfg.teacher.target_dim),
            "trained_layer": int(cfg.backbone.layer_idx),
            "load_s": round(t_load, 2),
            "eval_s": round(t_eval, 2),
            **metrics,
            "pred_report": pred_rep,
            "target_report": tgt_rep,
        }
        results.append(row)
        print(
            f"[head-eval] {exp}: r2={row['r2']:+.3f} loss={row['loss']:.4f} "
            f"mae={row['mean_abs_err']:.4f} cosine={row['cosine_mean']:+.3f} "
            f"pred[{_fmt_report(pred_rep)}] tgt[{_fmt_report(tgt_rep)}]"
        )

        del teacher
        del student
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (args.out / "head_eval.json").write_text(json.dumps(results, indent=2))
    header = "experiment,teacher_kind,loss_kind,target_dim,trained_layer,n_eval,n_dropped,target_var,target_constant,loss,r2,mean_abs_err,cosine_mean,load_s,eval_s\n"
    lines = [header]
    for row in results:
        lines.append(
            f"{row['experiment']},{row['teacher_kind']},{row['loss_kind']},"
            f"{row['target_dim']},{row['trained_layer']},{row['n_eval']},{row['n_dropped']},"
            f"{row['target_var']},{row['target_constant']},{row['loss']},{row['r2']},"
            f"{row['mean_abs_err']},{row['cosine_mean']},{row['load_s']},{row['eval_s']}\n"
        )
    (args.out / "head_eval.csv").write_text("".join(lines))
    print(f"[done] wrote {args.out / 'head_eval.json'} and {args.out / 'head_eval.csv'}")


if __name__ == "__main__":
    main()