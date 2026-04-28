"""Cross-prediction probe matrix.

For every pair (i, j) over the finetuned SDM_i and the matching teacher
representations, fit a linear probe from SDM_i's chunk-level latents to
teacher_j's chunk-level targets and record R^2 on a held-out split.

Default data: 100 utterances from ``mythicinfinity/libritts dev.clean``.
Small probe set deliberately — this is a sanity check on the factor
groupings, not a precision benchmark. Switch to a larger split or to
multilingual data once the matrix shape stabilises.

Run on a single TPU host (or any GPU/CPU box). Teacher targets and
backbone features are computed once each, then the probes themselves are
linear sklearn fits on numpy arrays.

Usage:
    uv run python scripts/run_linear_probes.py \\
        --consolidated checkpoints/consolidated \\
        --probe-utterances 100 \\
        --layer-sweep \\
        --out runs/probes/v0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

try:
    from sdm.data.streaming_emilia import EmiliaConfig, iter_chunks
    from sdm.data.teachers import build_teacher
    from sdm.modeling.distill_model import load_backbone
    from sdm.train.xla_utils import get_device
    _SDM_READY = True
except ImportError as e:  # pragma: no cover
    _SDM_READY = False
    _SDM_IMPORT_ERROR = repr(e)


EXPERIMENT_TO_FINETUNE_CONFIG = {
    "sdm-xlsr": "configs/finetune_xlsr_fairseq.yaml",
    "sdm-xlsr-fairseq": "configs/finetune_xlsr_fairseq.yaml",
    "sdm-dvector": "configs/finetune_dvector.yaml",
    "sdm-wespeaker": "configs/finetune_wespeaker.yaml",
    "sdm-pitch": "configs/finetune_pitch.yaml",
    "sdm-mpm": "configs/finetune_mpm.yaml",
    "sdm-speaking-rate": "configs/finetune_speaking_rate.yaml",
    "sdm-w2v2-asr": "configs/finetune_w2v2_asr.yaml",
    "sdm-mwhisper": "configs/finetune_mwhisper.yaml",
    "sdm-emotion2vec": "configs/finetune_emotion2vec.yaml",
}


def collect_held_out(
    repo_id: str,
    split: str,
    n_utt: int,
    *,
    config_name: str | None = None,
    seed: int = 42,
) -> list[dict]:
    """Pull a fixed slice of the held-out dataset for probing.

    Reuses the streaming Emilia loader (which is generic over HF audio
    datasets — see ``_extract_audio``). For LibriTTS the records arrive with
    standard ``{"audio": {"array", "sampling_rate"}}`` columns; for Emilia
    they arrive as WebDataset shards. ``iter_chunks`` handles both.
    """
    cfg = EmiliaConfig(
        repo_id=repo_id,
        config_name=config_name,
        split=split,
        streaming=True,
        shuffle_buffer=0,        # deterministic order; we just want the first N
        seed=seed,
        sample_rate=16000,
        chunk_seconds=1.0,
        max_chunks=32,
    )
    return list(iter_chunks(cfg, take=n_utt))


def _stack_batch(batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor, dict]:
    audio = torch.stack([r["audio"] for r in batch], dim=0)             # (B, N_chunks, T)
    mask = torch.stack([r["chunk_mask"] for r in batch], dim=0).bool()  # (B, N_chunks)
    ctx = {
        "texts": [r.get("text") for r in batch],
        "languages": [r.get("language") for r in batch],
        "ids": [r.get("id") for r in batch],
    }
    return audio, mask, ctx


def _flatten_valid(features: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """``features``: (B, N_chunks, D); ``mask``: (B, N_chunks). Return (n_valid, D)."""
    valid = features[mask]
    return valid.detach().to(torch.float32).cpu().numpy()


def encode_with_backbone(
    backbone, held_out: list[dict], *, layer: int, device: torch.device, batch_size: int
) -> np.ndarray:
    feats: list[np.ndarray] = []
    backbone.eval()
    for start in range(0, len(held_out), batch_size):
        batch = held_out[start : start + batch_size]
        audio, mask, _ = _stack_batch(batch)
        with torch.no_grad():
            h = backbone.encode(audio.to(device), layer=layer)        # (B, N_chunks, H)
        feats.append(_flatten_valid(h, mask.to(h.device)))
    return np.vstack(feats) if feats else np.zeros((0, 0), dtype=np.float32)


def teacher_targets(
    teacher, held_out: list[dict], *, device: torch.device, batch_size: int
) -> np.ndarray:
    out: list[np.ndarray] = []
    for start in range(0, len(held_out), batch_size):
        batch = held_out[start : start + batch_size]
        audio, mask, ctx = _stack_batch(batch)
        with torch.no_grad():
            y = teacher(audio.to(device), chunk_mask=mask.to(device), **ctx)  # (B, N_chunks, D)
        out.append(_flatten_valid(y, mask.to(y.device)))
    return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)


def _nonfinite_report(arr: np.ndarray) -> dict:
    """Count NaN / +Inf / -Inf cells and the rows containing any of them."""
    flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(-1, 1)
    n_nan = int(np.isnan(flat).sum())
    n_posinf = int(np.isposinf(flat).sum())
    n_neginf = int(np.isneginf(flat).sum())
    bad_rows = int((~np.isfinite(flat).all(axis=1)).sum())
    return {
        "n_nan": n_nan,
        "n_posinf": n_posinf,
        "n_neginf": n_neginf,
        "n_rows_bad": bad_rows,
        "n_rows_total": int(flat.shape[0]),
    }


def _fmt_report(rep: dict) -> str:
    return (
        f"NaN={rep['n_nan']} +Inf={rep['n_posinf']} -Inf={rep['n_neginf']} "
        f"bad_rows={rep['n_rows_bad']}/{rep['n_rows_total']}"
    )


def fit_probe(x: np.ndarray, y: np.ndarray, *, alpha: float = 1.0) -> dict:
    # Drop rows where either side contains non-finite values. Some teachers
    # (e.g. pitch / speaking-rate) emit NaN for silent / undefined chunks and
    # the backbone can produce NaN/Inf if the fp16 forward overflows; sklearn
    # Ridge refuses to fit either.
    x_finite = np.isfinite(x).all(axis=1)
    y_finite = np.isfinite(y).reshape(y.shape[0], -1).all(axis=1)
    finite = x_finite & y_finite
    n_dropped_x_only = int((~x_finite & y_finite).sum())
    n_dropped_y_only = int((x_finite & ~y_finite).sum())
    n_dropped_both = int((~x_finite & ~y_finite).sum())
    n_dropped = int((~finite).sum())
    if n_dropped:
        x = x[finite]
        y = y[finite]
    if x.shape[0] < 4:
        return {
            "r2_train": float("nan"),
            "r2_test": float("nan"),
            "n_test": int(x.shape[0]),
            "n_dropped": n_dropped,
            "n_dropped_x_only": n_dropped_x_only,
            "n_dropped_y_only": n_dropped_y_only,
            "n_dropped_both": n_dropped_both,
        }
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=0)
    model = Ridge(alpha=alpha)
    model.fit(x_tr, y_tr)
    return {
        "r2_train": float(r2_score(y_tr, model.predict(x_tr), multioutput="uniform_average")),
        "r2_test": float(r2_score(y_te, model.predict(x_te), multioutput="uniform_average")),
        "n_test": int(len(x_te)),
        "n_dropped": n_dropped,
        "n_dropped_x_only": n_dropped_x_only,
        "n_dropped_y_only": n_dropped_y_only,
        "n_dropped_both": n_dropped_both,
    }


def _backbone_model_id(experiment: str) -> str:
    cfg_path = EXPERIMENT_TO_FINETUNE_CONFIG[experiment]
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    return cfg["backbone"]["model_id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--consolidated", type=Path, required=True,
                    help="output dir of scripts/consolidate_weights.py")
    ap.add_argument("--dataset", default="mythicinfinity/libritts",
                    help="HF dataset repo id for the held-out probe split")
    ap.add_argument("--dataset-config", default="dev",
                    help="HF dataset config name (e.g. dev/clean/other/all for libritts)")
    ap.add_argument("--split", default="dev.clean",
                    help="dataset split (e.g. dev.clean / test.clean / train)")
    ap.add_argument("--probe-utterances", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="utterances per forward pass")
    ap.add_argument("--layer-sweep", action="store_true",
                    help="probe every backbone layer; otherwise just the configured one")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    if not _SDM_READY:
        raise SystemExit(
            "sdm runtime imports failed; cannot run probes. Last error: "
            + _SDM_IMPORT_ERROR
        )

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.consolidated / "manifest.json").read_text())
    experiments = list(manifest)
    device = get_device()
    print(f"[probe] device={device}  experiments={experiments}")

    print(f"[probe] loading {args.probe_utterances} utterances from "
          f"{args.dataset}[{args.dataset_config}]:{args.split}")
    held_out = collect_held_out(
        args.dataset, args.split, args.probe_utterances, config_name=args.dataset_config
    )
    print(f"[probe] held_out={len(held_out)} utterances "
          f"({sum(int(r['n_chunks']) for r in held_out)} valid chunks)")

    # Pre-compute teacher targets once (shared across SDM_i). Build, run, drop.
    target_matrix: dict[str, np.ndarray] = {}
    teacher_health: dict[str, dict] = {}
    for exp in experiments:
        cfg = yaml.safe_load(Path(EXPERIMENT_TO_FINETUNE_CONFIG[exp]).read_text())
        teacher = build_teacher(cfg["teacher"], device=device)
        target_matrix[exp] = teacher_targets(teacher, held_out, device=device, batch_size=args.batch_size)
        rep = _nonfinite_report(target_matrix[exp])
        teacher_health[exp] = rep
        print(f"[probe] teacher[{exp}] -> shape={target_matrix[exp].shape}  {_fmt_report(rep)}")
        if rep["n_rows_bad"]:
            print(
                f"        ^ {rep['n_rows_bad']} non-finite chunks in target; "
                f"likely silent/undefined (pitch=NaN on unvoiced; speaking_rate "
                f"can NaN on silence; ssl teachers can overflow in fp16)."
            )
        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Sweep latents per (sdm_exp, layer); fit probes against every target.
    backbone_model_ids: dict[str, str] = {}
    backbone_health: dict[tuple[str, int], dict] = {}
    results = []
    for sdm_exp in experiments:
        ckpt = args.consolidated / sdm_exp / "backbone.pt"
        backbone = load_backbone(ckpt, model_id=_backbone_model_id(sdm_exp))
        actual_id = getattr(backbone.backbone.config, "_name_or_path", None) or \
            f"hidden_size={backbone.backbone.config.hidden_size}"
        configured_id = _backbone_model_id(sdm_exp)
        backbone_model_ids[sdm_exp] = str(actual_id)
        if str(actual_id) != configured_id:
            print(
                f"[probe] WARNING: {sdm_exp} checkpoint backbone is {actual_id!r} "
                f"but YAML specifies {configured_id!r}"
            )
        backbone.to(device)
        layer_count = backbone.num_hidden_layers
        layers = range(layer_count + 1) if args.layer_sweep else [layer_count]
        for layer in layers:
            x = encode_with_backbone(backbone, held_out, layer=layer, device=device,
                                     batch_size=args.batch_size)
            x_rep = _nonfinite_report(x)
            backbone_health[(sdm_exp, int(layer))] = x_rep
            if x_rep["n_rows_bad"]:
                print(
                    f"[probe] {sdm_exp} L{layer}: backbone latents have "
                    f"{_fmt_report(x_rep)}"
                )
            for tgt_exp, y in target_matrix.items():
                if x.shape[0] != y.shape[0]:
                    raise RuntimeError(
                        f"chunk count mismatch: {sdm_exp}@L{layer} x={x.shape} vs {tgt_exp} y={y.shape}"
                    )
                probe = fit_probe(x, y)
                results.append({
                    "sdm": sdm_exp,
                    "layer": int(layer),
                    "target": tgt_exp,
                    **probe,
                })
                print(
                    f"{sdm_exp:>20s} L{layer:>2}  ->  {tgt_exp:<20s}  "
                    f"R2_test={probe['r2_test']:+.3f}"
                )
        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (args.out / "matrix.json").write_text(json.dumps(results, indent=2))
    print(f"[done] {len(results)} probes -> {args.out / 'matrix.json'}")

    # Summary of where non-finite values came from.
    bad_teachers = {exp: r for exp, r in teacher_health.items() if r["n_rows_bad"]}
    if bad_teachers:
        print("[probe] teachers with non-finite chunks:")
        for exp, r in bad_teachers.items():
            print(f"        {exp:>20s}  {_fmt_report(r)}")
    bad_backbones = {k: r for k, r in backbone_health.items() if r["n_rows_bad"]}
    if bad_backbones:
        print("[probe] backbones with non-finite latents:")
        for (exp, layer), r in bad_backbones.items():
            print(f"        {exp:>20s} L{layer:<2}  {_fmt_report(r)}")
    if not bad_teachers and not bad_backbones:
        print("[probe] no non-finite values detected anywhere.")

    # Summary of any backbone/yaml mismatches detected during the sweep.
    mismatched = {
        exp: actual
        for exp, actual in backbone_model_ids.items()
        if actual != _backbone_model_id(exp)
    }
    if mismatched:
        print("[probe] backbone/YAML mismatches:")
        for exp, actual in mismatched.items():
            print(f"        {exp:>20s}  ckpt={actual}  yaml={_backbone_model_id(exp)}")

    if args.wandb:
        import wandb  # noqa: PLC0415

        wandb.init(project="sdm", name="cross-probe-matrix", job_type="probe")
        wandb.log({"results": wandb.Table(columns=list(results[0]), data=[list(r.values()) for r in results])})
        wandb.finish()


if __name__ == "__main__":
    main()
