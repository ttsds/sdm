"""Pull the finetuned mHuBERT backbones from each experiment's GCS bucket and
stage them locally for the cross-prediction probe study.

Run on the same box you'll later run `scripts/run_linear_probes.py` from.

Usage:
    uv run python scripts/consolidate_weights.py \\
        --experiments sdm-xlsr sdm-dvector sdm-wespeaker sdm-pitch \\
                      sdm-mpm sdm-allosaurus sdm-w2v2-asr sdm-mwhisper \\
        --out checkpoints/consolidated
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import torch


def gcs_pull(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gsutil", "-q", "cp", src, str(dst)],
        check=True,
    )


def strip_to_backbone(state: dict) -> dict:
    """Keep only the backbone weights; drop the per-experiment head so the 8
    consolidated checkpoints are bit-for-bit comparable on backbone-only probes.
    """
    sd = state.get("model", state)
    return {k: v for k, v in sd.items() if not k.startswith("heads.")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", required=True)
    ap.add_argument("--ckpt-name", default="final.pt", help="filename inside each gs://sdm-ckpts/<exp>/ dir")
    ap.add_argument("--bucket", default="gs://sdm-ckpts")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep-heads", action="store_true", help="also retain per-experiment heads")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for exp in args.experiments:
        src = f"{args.bucket}/{exp}/{args.ckpt_name}"
        local = args.out / exp / args.ckpt_name
        print(f"[pull] {src} -> {local}")
        gcs_pull(src, local)

        state = torch.load(local, map_location="cpu")
        backbone = state if args.keep_heads else strip_to_backbone(state)
        backbone_path = args.out / exp / "backbone.pt"
        torch.save({"model": backbone, "experiment": exp}, backbone_path)
        manifest[exp] = {
            "ckpt": str(local),
            "backbone": str(backbone_path),
            "n_backbone_params": sum(v.numel() for v in backbone.values() if torch.is_tensor(v)),
        }
        if not args.keep_heads:
            local.unlink()

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] manifest at {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
