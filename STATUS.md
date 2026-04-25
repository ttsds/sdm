# Sdm — Status

Last updated: 2026-04-25.
Plan file: `~/.claude/plans/this-is-a-new-curried-badger.md`.
Repo: https://github.com/ttsds/sdm. TPU layout: [TPU_RESOURCES.md](TPU_RESOURCES.md).

## What is here

A unified speech-representation transformer (DeBERTa-v3 over NeuCodec FSQ tokens)
intended to replace TTSDS2's ~10-model teacher zoo with a single backbone plus
per-factor finetuned variants. Trained on `neuphonic/emilia-yodas-english-neucodec`
(78k h, 30M utterances) on TPU v4-32 / v4-64 via PyTorch/XLA. The wristband
Gaussian loss from `mvparakhin/ml-tidbits` is wired in as a toggleable ablation.

## Done

| Component | File(s) | Notes |
|---|---|---|
| Project skeleton | `pyproject.toml`, `.gitignore`, package layout | uv-managed; `teachers` and `neucodec` extras declared as conflicting (incompatible numpy ranges) |
| Streaming dataset loader | `sdm/data/neucodec_dataset.py` | FSQ codes shifted past 4 reserved special tokens (`PAD`/`CLS`/`SEP`/`MASK`) |
| Vocab/frame-rate probe | `scripts/probe_neucodec.py` | Run once before fixing pretrain config |
| DeBERTa-v3 backbone | `sdm/modeling/deberta_neucodec.py` | `SDM_BASE` (12L/768H) and `SDM_SMALL` (6L/384H) presets |
| MLM masking | `sdm/losses/mlm.py` | 80/10/10 BERT recipe restricted to FSQ-token positions |
| Pretraining loop | `sdm/train/pretrain.py`, `sdm/train/run_pretrain.py` | Single-host smoke test path + YAML-driven entry; cosine LR + warmup + grad-accum + checkpointing |
| TPU/XLA wrappers | `sdm/train/xla_utils.py` | `mark_step`, FSDP-via-XLA, master-only checkpoint save; degrade cleanly on CPU/GPU |
| TPU launcher | `scripts/launch_tpu_pretrain.sh` | Sets `PJRT_DEVICE=TPU`, `XLA_USE_BF16=1` |
| Pretrain config | `configs/pretrain_base.yaml` | `batch_size=16` per-device, `total_steps=250000`, `warmup=4000` |
| Teacher target framework | `sdm/data/teacher_cache.py`, `scripts/extract_teacher_targets.py` | Per-utterance `(wav, sr) -> np.ndarray` adapters; sharded `.npz` output |
| Teacher cache reader | `sdm/data/teacher_dataset.py` | Pairs codes with targets; CLS-prefix-aware sequence alignment |
| Distillation heads | `sdm/modeling/distillation_heads.py` | Sequence and pooled head modes; weighted multi-teacher MSE loss |
| Distillation finetune | `sdm/train/finetune.py` | Same XLA-aware loop pattern as pretrain; loads pretrained ckpt with `strict=False` |
| Finetune configs | `configs/finetune_{generic,intelligibility}.yaml` + `..._gaussian_ablation.yaml` | One config per factor, ablation flag at `train.gaussian_loss.enabled` |
| TTSDS adapter | `sdm/eval/ttsds_factor_eval.py` | Subclasses upstream `Benchmark`; reuses upstream `wasserstein_distance` / `frechet_distance` unchanged |
| End-to-end eval CLI | `sdm/eval/correlate_with_mos.py` | Iterates `<dir>/<domain>/<system>/*.wav`, computes Spearman vs MOS/CMOS/SMOS |
| Wristband Gaussian loss | `sdm/losses/wristband_gaussian.py` | Direct port of `C_WristbandGaussianLoss` (MIT, attribution preserved) |
| GCS / spot-resume plumbing | `sdm/train/io.py`, `sdm/train/preempt.py`, updated `scripts/launch_tpu_pretrain.sh` | `gs://` ckpt paths via `gsutil`; SIGTERM flushes `latest.pt`; outer retry loop |
| W&B tracking | `sdm/train/wandb_utils.py`, `wandb` block in every config | Master-only init; lazy import; `WANDB_DISABLED=true` opt-out |
| Pretrain ablation config | `configs/pretrain_base_gaussian.yaml` | Same hyperparams + `gaussian_loss.enabled: true` for the v6e/use1 slice |
| TPU resource inventory | `TPU_RESOURCES.md` | TRC pools → roles, GCS bucket layout, `queued-resources` sketch |
| Tests | `tests/test_*.py` | 26 passing after rename |

## Partial / stubbed

### Teacher coverage (the largest remaining chunk)

`sdm/data/teacher_cache.py` ports the teachers that share a clean SSL/encoder
API. The following are still `_todo(...)` stubs that raise with concrete
upstream paths:

| Teacher | Factor | Upstream to port |
|---|---|---|
| WeSpeaker | speaker | `/tmp/ttsds/src/ttsds/benchmarks/speaker/wespeaker.py` (uses `pyannote.audio.Inference` over a temp wav) |
| D-Vector | speaker | `/tmp/ttsds/src/ttsds/benchmarks/speaker/dvector.py` (windowed sliding embeddings) |
| MPM | prosody | `/tmp/ttsds/src/ttsds/benchmarks/prosody/mpm.py` + `ttsds/util/mpm.py` (custom transformer; layer 7 activations) |
| Allosaurus | prosody | `/tmp/ttsds/src/ttsds/benchmarks/prosody/allosaurus.py` (phone count -> speaking rate) |
| HuBERT-token-rate | prosody | `/tmp/ttsds/src/ttsds/benchmarks/prosody/hubert_token.py` (KMeans on HuBERT tokens) |
| XLSR-53 | generic + intelligibility | `/tmp/ttsds/src/ttsds/benchmarks/generic/wav2vec2_xlsr.py` (mostly drop-in to `_SSLTeacher`) |
| mHuBERT-147 | generic | `/tmp/ttsds/src/ttsds/benchmarks/generic/mhubert.py` (drop-in to `_SSLTeacher`) |
| wav2vec2-base-960h | intelligibility | `/tmp/ttsds/src/ttsds/benchmarks/intelligibility/w2v2_activations.py` (last-hidden mean-pool) |

Until these are ported, `sdm-speaker` and `sdm-prosody` finetunes cannot
run end-to-end. `sdm-generic` (HuBERT/WavLM/wav2vec2) and
`sdm-intelligibility` (Whisper) are fully wired today.

### NeuCodec encode API

`sdm/eval/ttsds_factor_eval.py:_SdmEncoder.wav_to_codes` calls
`codec.encode_code` with a fallback to `codec.encode`. The exact API name on
the installed `neucodec` build needs to be confirmed before the first
end-to-end eval run.

### Vocab size and frame rate

`configs/*.yaml` hard-code `fsq_vocab_size: 65536` as a placeholder. Run
`uv run python scripts/probe_neucodec.py --num-records 1000` and update the
configs once the real numbers are known.

### Local Python 3.14 quirks

PyTorch on Python 3.14 emits a `torch.jit.script` deprecation warning. Harmless
today, but a future torch upgrade may break this; pin against a known-good
torch version when locking the TPU image.

## Outstanding work (in suggested order)

1. **Probe the dataset.** Run `scripts/probe_neucodec.py`, then update
   `fsq_vocab_size` and `max_position_embeddings` in all configs.
2. **Port the remaining teachers** (table above). Each is small if you mirror
   the per-utterance internals of the corresponding `_get_distribution`. Add
   one test per teacher that confirms the output shape on a 1-second random
   waveform (gated behind `pytest.importorskip` for heavy deps).
3. **Build a small teacher cache.** `extract_teacher_targets.py --factor generic
   --num-records 50000 --shard-size 1000 --out teacher_cache/generic` on a
   GPU box. Estimate ~6 h with a single A100; parallelise across hosts if
   needed.
4. **Run pretrain on TPU.** `bash scripts/launch_tpu_pretrain.sh
   configs/pretrain_base.yaml`. Watch for FSDP-via-XLA shape errors on the
   first compile; if they appear, drop `fsdp: true` while iterating.
5. **Finetune `sdm-generic`.** `python -m sdm.train.finetune --config
   configs/finetune_generic.yaml` once the cache exists. Confirm distillation
   MSE drops monotonically.
6. **End-to-end correlation.** `python -m sdm.eval.correlate_with_mos
   --sdm-ckpt checkpoints/finetune_generic/final.pt --sdm-config
   configs/finetune_generic.yaml --factor generic --head hubert
   --listening-test-dir <local copy of hf.co/datasets/ttsds/listening_test>
   --out runs/eval/generic.csv`. Target: Spearman ρ within 0.05 of the
   original HuBERT row of the TTSDS2 paper Table 3 on Clean.
7. **Wristband Gaussian ablation.** Repeat step 5 with
   `configs/finetune_generic_gaussian_ablation.yaml`. Compare ρ deltas and
   inspect the W2 distance distributions before/after.
8. **Speaker + Prosody.** Once teachers are ported, repeat 3-7 for those
   factors.
9. **Multilingual.** Out of scope today; the existing TTSDS2 multilingual
   pipeline can wrap sdm variants once we add an mHuBERT-style mC variant.

## How to run

```bash
# Install dev deps (no torch-xla, no teachers).
uv sync --extra dev

# Tests.
uv run pytest tests/

# Probe dataset (needs `neucodec` extra on a numpy>=2 host).
uv sync --extra neucodec --extra dev
uv run python scripts/probe_neucodec.py --num-records 1000

# Build teacher cache (needs `teachers` extra on a numpy<2 host -- separate venv).
uv sync --extra teachers --extra dev
uv run python scripts/extract_teacher_targets.py --factor generic \
    --num-records 50000 --shard-size 1000 --out teacher_cache/generic

# Pretrain on TPU.
bash scripts/launch_tpu_pretrain.sh configs/pretrain_base.yaml

# Finetune generic.
uv run python -m sdm.train.finetune --config configs/finetune_generic.yaml
```

## Reference repos (cloned at)

- `/tmp/ttsds`     — https://github.com/ttsds/ttsds (Benchmark ABC, distance utilities, teacher loaders)
- `/tmp/ml-tidbits` — https://github.com/mvparakhin/ml-tidbits (`C_WristbandGaussianLoss` source)
