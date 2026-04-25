"""End-to-end sdm evaluation against the public TTSDS2 listening-test data.

Reproduces the relevant rows of Table 3 in the TTSDS2 paper, swapping the
original teacher benchmarks for sdm's finetuned variants. Requires the
`teachers` extra (provides the upstream `ttsds` package).

Run on a CPU/GPU host. Output is a CSV with one row per (system, domain)
plus a final Spearman-correlation summary.

Usage:
    uv run python -m sdm.eval.correlate_with_mos \\
        --sdm-ckpt checkpoints/finetune_generic/final.pt \\
        --factor generic --head hubert \\
        --listening-test-dir /path/to/ttsds-listening-test \\
        --out runs/eval/generic.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sdm.eval.ttsds_factor_eval import SdmBenchmarkConfig, make_sdm_benchmark
from sdm.modeling.deberta_neucodec import SdmConfig
from sdm.modeling.distillation_heads import HeadSpec


def _load_specs(path: Path) -> list[HeadSpec]:
    """Read head_specs from a finetune YAML so the adapter can rebuild them."""
    import yaml  # noqa: PLC0415

    raw = yaml.safe_load(path.read_text())
    return [HeadSpec(**h) for h in raw["heads"]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sdm-ckpt", required=True)
    p.add_argument("--sdm-config", required=True, help="finetune YAML used to train this ckpt")
    p.add_argument("--factor", required=True, choices=("generic", "speaker", "prosody", "intelligibility"))
    p.add_argument("--head", required=True, help="teacher head name, e.g. 'hubert'")
    p.add_argument("--listening-test-dir", required=True, type=Path)
    p.add_argument("--mos-csv", default=None, help="public MOS labels (default: read from listening-test-dir)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    import yaml  # noqa: PLC0415

    raw = yaml.safe_load(Path(args.sdm_config).read_text())
    model_cfg = SdmConfig(**raw["model"])
    head_specs = _load_specs(Path(args.sdm_config))

    bench_cfg = SdmBenchmarkConfig(
        factor=args.factor,
        head_name=args.head,
        checkpoint_path=args.sdm_ckpt,
        model_cfg=model_cfg,
        head_specs=head_specs,
        device=args.device,
        max_length=model_cfg.max_position_embeddings,
    )
    bench = make_sdm_benchmark(bench_cfg)

    # Datasets are organised as <listening_test_dir>/<domain>/<system>/*.wav.
    # The ground truth is at <listening_test_dir>/<domain>/ground_truth/*.wav.
    from ttsds.ttsds import BenchmarkSuite  # noqa: PLC0415
    from ttsds.util.dataset import DirectoryDataset  # noqa: PLC0415

    domains = sorted(d.name for d in args.listening_test_dir.iterdir() if d.is_dir())
    rows = []
    for domain in domains:
        gt_dir = args.listening_test_dir / domain / "ground_truth"
        if not gt_dir.exists():
            continue
        ref = DirectoryDataset(str(gt_dir), name=f"{domain}/gt", sample_rate=16000)
        for sys_dir in sorted((args.listening_test_dir / domain).iterdir()):
            if sys_dir.name == "ground_truth" or not sys_dir.is_dir():
                continue
            ds = DirectoryDataset(str(sys_dir), name=f"{domain}/{sys_dir.name}", sample_rate=16000)
            suite = BenchmarkSuite(
                datasets=[ds],
                reference_datasets=[ref],
                benchmarks={f"sdm_{args.factor}_{args.head}": bench},
                device=args.device,
            )
            df = suite.run()
            score = float(df["score"].mean())
            rows.append({"domain": domain, "system": sys_dir.name, "ttsds": score})
            print(f"{domain:>8s}/{sys_dir.name:<20s}  TTSDS={score:.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd  # noqa: PLC0415

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    if args.mos_csv:
        from scipy.stats import spearmanr  # noqa: PLC0415

        labels = pd.read_csv(args.mos_csv)
        merged = df.merge(labels, on=["domain", "system"], how="inner")
        summary = {}
        for col in ("MOS", "CMOS", "SMOS"):
            if col in merged.columns:
                rho, p = spearmanr(merged["ttsds"], merged[col])
                summary[col] = {"rho": float(rho), "p": float(p)}
                print(f"  Spearman vs {col}: rho={rho:+.3f} p={p:.3g}")
        (args.out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
