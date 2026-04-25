"""Probe `neuphonic/emilia-yodas-english-neucodec` to determine FSQ vocab size
and token frame rate. Run once before fixing the pretraining config.

Usage:
    python scripts/probe_neucodec.py --num-records 1000
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-records", type=int, default=1000)
    parser.add_argument("--repo-id", default="neuphonic/emilia-yodas-english-neucodec")
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.repo_id, split="train", streaming=True)

    max_code = -1
    min_code = 10**9
    counts: list[int] = []
    rates: list[float] = []
    durations: list[float] = []

    for i, rec in enumerate(ds):
        if i >= args.num_records:
            break
        codes = rec["codes"]
        n = len(codes)
        d = float(rec["duration"])
        counts.append(n)
        durations.append(d)
        if d > 0:
            rates.append(n / d)
        max_code = max(max_code, int(np.max(codes)))
        min_code = min(min_code, int(np.min(codes)))

    print(f"Sampled {len(counts)} records.")
    print(f"FSQ code value range: [{min_code}, {max_code}]")
    print(f"Implied min vocab size: {max_code + 1}")
    print(f"Tokens per record: mean={np.mean(counts):.1f} max={np.max(counts)}")
    print(f"Duration s: mean={np.mean(durations):.2f} max={np.max(durations):.2f}")
    print(f"Frame rate (tokens/s): mean={np.mean(rates):.2f} median={np.median(rates):.2f}")


if __name__ == "__main__":
    main()
