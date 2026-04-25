"""Run teacher models over a streamed Emilia-YODAS subset, caching their
outputs to sharded `.npz` files for later distillation training.

Run on a CPU/GPU host with the `teachers` and `neucodec` extras installed.
Avoid running on a TPU host (numpy version conflicts; teachers are slow on CPU).

Usage:
    uv run python scripts/extract_teacher_targets.py \\
        --factor generic --num-records 100000 \\
        --shard-size 1000 --out teacher_cache/generic
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from sdm.data.teacher_cache import decode_neucodec, get_factor_teachers
from sdm.dotenv import load_dotenv


def _shard_path(out: Path, shard_idx: int) -> Path:
    return out / f"shard-{shard_idx:06d}.npz"


def _flush(out: Path, shard_idx: int, buffer: dict[str, list]) -> None:
    arrays: dict[str, np.ndarray | list] = {}
    arrays["ids"] = np.array(buffer["ids"], dtype=object)
    arrays["codes_lengths"] = np.array([len(c) for c in buffer["codes"]], dtype=np.int32)
    arrays["codes_concat"] = np.concatenate(
        [np.asarray(c, dtype=np.int32) for c in buffer["codes"]]
    )
    for key, vals in buffer.items():
        if key in {"ids", "codes"}:
            continue
        # Variable-length per utterance -> store concatenated + lengths
        lengths = np.array([np.asarray(v).shape[0] for v in vals], dtype=np.int32)
        flat = np.concatenate([np.asarray(v).reshape(lengths[i], -1) for i, v in enumerate(vals)])
        arrays[f"{key}_concat"] = flat
        arrays[f"{key}_lengths"] = lengths
    np.savez_compressed(_shard_path(out, shard_idx), **arrays)


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--factor", required=True, choices=("generic", "speaker", "prosody", "intelligibility"))
    p.add_argument("--num-records", type=int, default=10_000)
    p.add_argument("--shard-size", type=int, default=500)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--repo-id", default="neuphonic/emilia-yodas-english-neucodec")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from neucodec import NeuCodec

    print(f"loading neucodec...")
    codec = NeuCodec.from_pretrained("neuphonic/neucodec").eval()

    print(f"loading teachers for factor={args.factor}...")
    teachers = get_factor_teachers(args.factor, device=args.device)
    if not teachers:
        raise SystemExit(f"No teachers available for factor {args.factor}")
    print(f"  -> {list(teachers)}")

    ds = load_dataset(args.repo_id, split="train", streaming=True)

    buffer: dict[str, list] = {"ids": [], "codes": [], **{name: [] for name in teachers}}
    shard_idx = 0
    t0 = time.time()

    for i, rec in enumerate(ds):
        if i >= args.num_records:
            break
        codes = rec["codes"]
        wav, sr = decode_neucodec(codes, model=codec)
        buffer["ids"].append(rec["id"])
        buffer["codes"].append(codes)
        for name, teacher in teachers.items():
            buffer[name].append(teacher(wav, sr))

        if len(buffer["ids"]) >= args.shard_size:
            _flush(args.out, shard_idx, buffer)
            elapsed = time.time() - t0
            print(
                f"shard {shard_idx} flushed ({(shard_idx + 1) * args.shard_size} records, "
                f"{elapsed:.1f}s, {(shard_idx + 1) * args.shard_size / elapsed:.2f} rec/s)"
            )
            shard_idx += 1
            buffer = {"ids": [], "codes": [], **{n: [] for n in teachers}}

    if buffer["ids"]:
        _flush(args.out, shard_idx, buffer)
        print(f"final shard {shard_idx} flushed")

    print(f"done: wrote {shard_idx + 1} shards to {args.out}")


if __name__ == "__main__":
    main()
