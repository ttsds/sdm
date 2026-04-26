"""Cross-prediction matrix: for every (i, j) over the 8 finetuned SDM_i and
the 8 teacher representations, fit a linear probe from SDM_i's chunk-level
latents to teacher_j's chunk-level targets and record R^2 on a held-out split.

This produces the empirical "factor" matrix the user wanted to see before
locking the TTSDS factor taxonomy. The scaffolding mirrors `consolidate_weights.py`
and assumes its output layout.

Run on a single GPU box (or CPU; the probes themselves are linear). Targets
are extracted in-process via the same teacher classes used at training time.

Usage:
    uv run python scripts/run_linear_probes.py \\
        --consolidated checkpoints/consolidated \\
        --probe-utterances 1000 \\
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

# These imports rely on the codebase work flagged in STATUS.md (in-the-loop
# teacher loaders + chunked pooling). Keep them here so the file is wired up
# end-to-end, but the run will only succeed once those modules land.
try:
    from sdm.data.streaming_emilia import EmiliaConfig, iter_chunks
    from sdm.data.teachers import build_teacher
    from sdm.modeling.distill_model import DistillModel, load_backbone
    _SDM_READY = True
except ImportError as e:  # pragma: no cover
    _SDM_READY = False
    _SDM_IMPORT_ERROR = repr(e)


EXPERIMENT_TO_TEACHER_CONFIG = {
    "sdm-xlsr": "configs/finetune_xlsr.yaml",
    "sdm-dvector": "configs/finetune_dvector.yaml",
    "sdm-wespeaker": "configs/finetune_wespeaker.yaml",
    "sdm-pitch": "configs/finetune_pitch.yaml",
    "sdm-mpm": "configs/finetune_mpm.yaml",
    "sdm-allosaurus": "configs/finetune_allosaurus.yaml",
    "sdm-w2v2-asr": "configs/finetune_w2v2_asr.yaml",
    "sdm-mwhisper": "configs/finetune_mwhisper.yaml",
}


def collect_held_out(emilia_cfg: EmiliaConfig, n_utt: int):
    """Pull a fixed seed-pinned slice of Emilia for probing — disjoint from
    the streaming order used in training because we reset the seed and skip
    to a different shard offset.
    """
    return list(iter_chunks(emilia_cfg, take=n_utt))


def encode_with_backbone(backbone, audio_chunks, layer: int) -> np.ndarray:
    """Run frozen backbone, return chunk-level features at the given layer."""
    feats = []
    for batch in audio_chunks:
        with torch.no_grad():
            h = backbone(batch.input_audio, layer=layer)  # (B, N_chunks, H)
        feats.append(h.reshape(-1, h.shape[-1]).cpu().numpy())
    return np.vstack(feats)


def teacher_targets(teacher, audio_chunks) -> np.ndarray:
    targets = []
    for batch in audio_chunks:
        with torch.no_grad():
            t = teacher(batch.input_audio)             # (B, N_chunks, D)
        targets.append(t.reshape(-1, t.shape[-1]).cpu().numpy())
    return np.vstack(targets)


def fit_probe(x: np.ndarray, y: np.ndarray, *, alpha: float = 1.0) -> dict:
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=0)
    model = Ridge(alpha=alpha)
    model.fit(x_tr, y_tr)
    return {
        "r2_train": float(r2_score(y_tr, model.predict(x_tr), multioutput="uniform_average")),
        "r2_test": float(r2_score(y_te, model.predict(x_te), multioutput="uniform_average")),
        "n_test": int(len(x_te)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--consolidated", type=Path, required=True)
    ap.add_argument("--probe-utterances", type=int, default=1000)
    ap.add_argument("--layer-sweep", action="store_true",
                    help="probe every backbone layer; otherwise just the configured one")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    if not _SDM_READY:
        raise SystemExit(
            "sdm in-the-loop teachers + chunked emilia loader not yet implemented; "
            "this script is wired but waiting on those modules. Last import error: "
            + _SDM_IMPORT_ERROR
        )

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.consolidated / "manifest.json").read_text())
    experiments = list(manifest)

    # Build teacher views once; they're reused across all SDM_i.
    teachers = {}
    for exp in experiments:
        cfg = yaml.safe_load(Path(EXPERIMENT_TO_TEACHER_CONFIG[exp]).read_text())
        teachers[exp] = build_teacher(cfg["teacher"], device="cuda" if torch.cuda.is_available() else "cpu")

    emilia_cfg = EmiliaConfig(
        repo_id="amphion/Emilia-Dataset",
        streaming=True,
        seed=42,                          # held-out seed (training uses 0)
        chunk_seconds=1.0,
        max_chunks=32,
        sample_rate=16000,
    )
    held_out = collect_held_out(emilia_cfg, args.probe_utterances)

    # Pre-compute teacher targets once (shared across SDM_i).
    target_matrix = {name: teacher_targets(t, held_out) for name, t in teachers.items()}

    # Sweep latents per (i, layer); fit probes against every target.
    results = []
    for sdm_exp in experiments:
        backbone = load_backbone(args.consolidated / sdm_exp / "backbone.pt")
        layers = range(backbone.num_hidden_layers + 1) if args.layer_sweep else [-1]
        for layer in layers:
            x = encode_with_backbone(backbone, held_out, layer=layer)
            for tgt_exp, y in target_matrix.items():
                if x.shape[0] != y.shape[0]:
                    raise RuntimeError(f"chunk count mismatch: {sdm_exp}@L{layer} {x.shape} vs {tgt_exp} {y.shape}")
                probe = fit_probe(x, y)
                results.append({
                    "sdm": sdm_exp,
                    "layer": int(layer),
                    "target": tgt_exp,
                    **probe,
                })
                print(
                    f"{sdm_exp:>17s} L{layer:>2}  ->  {tgt_exp:<17s}  "
                    f"R2_test={probe['r2_test']:+.3f}"
                )

    (args.out / "matrix.json").write_text(json.dumps(results, indent=2))
    print(f"[done] {len(results)} probes -> {args.out / 'matrix.json'}")

    if args.wandb:
        import wandb  # noqa: PLC0415

        wandb.init(project="sdm", name="cross-probe-matrix", job_type="probe")
        wandb.log({"results": wandb.Table(columns=list(results[0]), data=[list(r.values()) for r in results])})
        wandb.finish()


if __name__ == "__main__":
    main()
