# SDM — Speech Distribution Model

One backbone, many checkpoints, distilled to reproduce the teacher representations that [TTSDS2](TTSDS2_paper.pdf) uses to score synthetic speech.

The headline idea: instead of running 10 independent teacher models at evaluation time (HuBERT, WavLM, wav2vec 2.0, Whisper, d-Vector, WeSpeaker, WORLD F0, MPM, Allosaurus, …), we train a single **XLS-R 300M** student to predict each teacher's representation on 1-second chunks of multilingual Emilia. Each finetune produces one checkpoint of the same architecture; downstream code can load whichever checkpoint matches the factor it cares about (Generic, Speaker, Prosody, Intelligibility).

## Quick links

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — pipeline, Teacher protocol, TTSDS2 factor taxonomy, XLA-bf16 safety choices.
- [`ROADMAP.md`](ROADMAP.md) — what's next: remaining 7 Teachers → cross-teacher probes → wristband-Gaussian regularizer.
- [`TPU_RESOURCES.md`](TPU_RESOURCES.md) — GCP project, TPU pool inventory, GCS bucket layout.
- [`TTSDS2_paper.pdf`](TTSDS2_paper.pdf) — the paper this project is downstream of.

## Repo layout

```
sdm/
  data/
    streaming_emilia.py     # multilingual Emilia → 1s chunks → fixed shapes
    teachers/
      __init__.py           # build_teacher dispatcher (keyed on cfg.teacher.kind)
      hf_ssl.py             # HF SSL teacher (mHuBERT-147 today)
  modeling/
    distill_model.py        # backbone wrapper; "fairseq_w2v2" / "tiny" / "hf" branches
    wav2vec2_fairseq.py     # in-tree XLS-R port (Fp32 GN/LN, baked pos weight_norm, SDPA)
  losses/
    wristband_gaussian.py   # Gaussian regularizer (kept for the future Wasserstein phase)
  train/
    run_distill.py          # entrypoint: python -m sdm.train.run_distill --config <yaml>
    xla_utils.py            # XLA / FSDP / multi-process plumbing
    io.py                   # local + gs:// checkpoint I/O
    preempt.py              # SIGTERM handling for spot TPUs
    wandb_utils.py          # lazy W&B integration
  dotenv.py                 # loads WANDB_API_KEY / HF token from .env
configs/
  finetune_xlsr_fairseq.yaml  # LIVE: distill mHuBERT-147 L8 → XLS-R 300M student
  finetune_tiny.yaml          # CPU debug variant (tiny in-tree transformer)
  finetune_dvector.yaml       # speaker — d-Vector torchscript (TTSDS bundle)
  finetune_wespeaker.yaml     # speaker — pyannote WeSpeaker ResNet34
  finetune_pitch.yaml         # prosody — WORLD F0 mean per chunk
  finetune_mpm.yaml           # prosody — Masked Prosody Model layer 7 (English-only)
  finetune_speaking_rate.yaml # prosody — phones/sec from G2P on the dataset transcripts
  finetune_w2v2_asr.yaml      # intelligibility — wav2vec2 ASR encoder
  finetune_mwhisper.yaml      # intelligibility — multilingual Whisper-small encoder
scripts/
  run_finetune.sh             # canonical TPU launcher; takes CONFIG=<path>
  _tpu_common.sh              # shared bootstrap (clone, uv sync, .env, env vars)
  consolidate_weights.py      # pulls per-experiment final.pt into a probe staging dir
  run_linear_probes.py        # cross-teacher R² matrix (probe phase)
  tpu/                        # GCP TPU provisioning helpers
tests/                        # pytest suite covering the live distill path
pyproject.toml                # uv-managed; deps split into base / teachers / tpu / tracking / dev
```

## Prerequisites

- Python ≥ 3.10, [`uv`](https://github.com/astral-sh/uv) for dependency management.
- A `.env` at the repo root with at least `HF_TOKEN=…` (and `WANDB_API_KEY=…` if W&B logging is enabled). See `sdm/dotenv.py` for the variables it reads.
- For TPU runs: a GCS bucket reachable from the slice (configs default to `gs://sdm-ckpts/<experiment>/`) and a baked TPU image with matching `torch` + `torch_xla` wheels.

```bash
uv sync --extra dev --extra tracking
```

The `teachers` extra installs heavyweight feature extractors (`ttsds`, `pyworld`, `wespeaker-unofficial`, …) and is only needed on a host that builds offline teacher caches — not on TPU training hosts.

## Running a finetune

**Local CPU smoke** (uses the tiny in-tree backbone — fast, no GPU/TPU needed):

```bash
uv run python -m sdm.train.run_distill --config configs/finetune_tiny.yaml
```

**Single-host GPU/CPU run of the live config**:

```bash
uv run python -m sdm.train.run_distill --config configs/finetune_xlsr_fairseq.yaml
```

**TPU (spot, with retry on preemption)**:

```bash
CONFIG=configs/finetune_xlsr_fairseq.yaml ./scripts/run_finetune.sh
```

The launcher sources `scripts/_tpu_common.sh` (clone + `uv sync` + `.env` + XLA env vars), then loops on `python -m sdm.train.run_distill`. SIGTERM on preemption flushes a checkpoint and exits 130; the loop relaunches and `resume_from_latest` in the YAML picks up where we left off.

## Tests

```bash
uv run pytest -q
```

Exercises the live path: streaming Emilia → chunk collate → fairseq XLS-R + HF SSL teacher → masked cosine + L1 loss → checkpoint I/O.

## Adding a new Teacher

See [`ARCHITECTURE.md#adding-a-teacher`](ARCHITECTURE.md#adding-a-teacher) for the full contract. Sketch:

1. Create `sdm/data/teachers/<kind>.py` exposing a `Teacher`-shaped class with `__call__(audio, chunk_mask) -> Tensor[B, N_chunks, D]`.
2. Register the `<kind>` string in `sdm/data/teachers/__init__.py`.
3. Drop the matching scaffolding YAML into action (it's already in `configs/finetune_<kind>.yaml`); update `target_dim`, `model_id`, `layer` if the implementation differs from the placeholder.
4. `CONFIG=configs/finetune_<kind>.yaml ./scripts/run_finetune.sh`.
