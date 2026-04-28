# SDM — Speech Distribution Model

One backbone, many checkpoints, distilled to reproduce the teacher representations that [TTSDS2](TTSDS2_paper.pdf) uses to score synthetic speech.

The headline idea: instead of running 9+ independent teacher models at evaluation time (HuBERT, WavLM, wav2vec 2.0, Whisper, d-Vector, WeSpeaker, WORLD F0, MPM, syllables/sec, emotion2vec, …), we train a single **XLS-R 300M** student to predict each teacher's representation on 1-second chunks of multilingual Emilia. Each finetune produces one checkpoint of the same architecture; downstream code can load whichever checkpoint matches the factor it cares about (Generic, Speaker, Prosody, Intelligibility, Emotion).

**Status: Phase 1 done.** All nine finetunes have trained 10 k steps on 10% of multilingual Emilia and produced `final.pt` checkpoints under `gs://sdm-ckpts/<experiment>/`. Active focus is the cross-teacher probe matrix (Phase 2) — see [`ROADMAP.md`](ROADMAP.md).

## Quick links

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — pipeline, Teacher protocol, TTSDS2 factor taxonomy, XLA-bf16 safety choices.
- [`ROADMAP.md`](ROADMAP.md) — Phase 1 done (9 finetunes); Phase 2 active (cross-teacher probes); Phase 3 (wristband Gaussian) queued.
- [`TPU_RESOURCES.md`](TPU_RESOURCES.md) — GCP project, TPU pool inventory, GCS bucket layout.
- [`.tpu/README.md`](.tpu/README.md) — per-experiment provisioning scripts (sourced from `.env`, no longer gitignored).
- [`TTSDS2_paper.pdf`](TTSDS2_paper.pdf) — the paper this project is downstream of.

## Repo layout

```
sdm/
  data/
    streaming_emilia.py     # multilingual Emilia → 1s chunks → fixed shapes
    teachers/               # one module per teacher kind, dispatched via build_teacher
      __init__.py           # registry keyed on cfg.teacher.kind
      hf_ssl.py             # mHuBERT-147 (and any HF SSL with output_hidden_states)
      hf_ctc.py             # wav2vec2 ASR encoder (intelligibility)
      whisper_encoder.py    # multilingual Whisper-small encoder
      dvector_torchscript.py
      wespeaker.py
      pyworld_f0.py         # 1-D scalar F0 mean per chunk
      mpm.py                # Masked Prosody Model L7 (English-only, 22.05 kHz)
      g2p_speaking_rate.py  # syllables/sec from transcript phonemization
      emotion2vec.py        # FunASR emotion2vec base
  modeling/
    distill_model.py        # backbone wrapper; "fairseq_w2v2" / "tiny" / "hf" branches
    wav2vec2_fairseq.py     # in-tree XLS-R port (Fp32 GN/LN, baked pos weight_norm, SDPA)
  losses/
    wristband_gaussian.py   # Gaussian regularizer (queued for Phase 3)
  train/
    run_distill.py          # entrypoint: python -m sdm.train.run_distill --config <yaml>
    xla_utils.py            # XLA / FSDP / multi-process plumbing
    io.py                   # local + gs:// checkpoint I/O
    preempt.py              # SIGTERM handling for spot TPUs
    wandb_utils.py          # lazy W&B integration
  dotenv.py                 # loads WANDB_API_KEY / HF token from .env
configs/
  finetune_xlsr.yaml          # generic — mHuBERT-147 L8 (the reference run)
  finetune_xlsr_fairseq.yaml  # generic variant with the in-tree fairseq XLS-R port (kept for parity)
  finetune_dvector.yaml       # speaker — d-Vector torchscript (TTSDS bundle)
  finetune_wespeaker.yaml     # speaker — wespeaker-unofficial multilingual SimAMResNet34
  finetune_pitch.yaml         # prosody — WORLD F0 mean per chunk
  finetune_mpm.yaml           # prosody — Masked Prosody Model layer 7
  finetune_speaking_rate.yaml # prosody — syllables/sec from G2P on transcripts
  finetune_w2v2_asr.yaml      # intelligibility — wav2vec2 ASR encoder
  finetune_mwhisper.yaml      # intelligibility — multilingual Whisper-small encoder
  finetune_emotion2vec.yaml   # emotion — FunASR emotion2vec base
  finetune_tiny.yaml          # CPU debug variant (tiny in-tree transformer)
scripts/
  run_finetune.sh             # canonical in-VM launcher; takes CONFIG=<path>
  run_probes.sh               # in-VM driver for Phase 2 (consolidate + cross-teacher probes)
  _tpu_common.sh              # shared bootstrap (clone, uv sync, .env, env vars)
  consolidate_weights.py      # pulls per-experiment final.pt into a probe staging dir
  run_linear_probes.py        # 9 × 9 cross-teacher R² matrix (Phase 2)
  tpu/                        # generic capacity-hunting helpers (bake_vm, sdm_tpu CLI, …)
.tpu/                         # per-experiment provisioning (committed; secrets via .env)
  _lib.sh                     # shared poll/create/launch helpers; sources .env
  provision_<short>.sh        # one per experiment; brings up VM, scp .env, tmux launch
  provision_probes.sh         # brings up sdm-probes and runs scripts/run_probes.sh
  provision_all.sh            # nohup fan-out across all 8 finetunes
  teardown.sh                 # gcloud delete for every known VM + kill local poll loops
  logs/                       # runtime poll-loop logs (gitignored)
tests/                        # pytest suite covering the live distill path
pyproject.toml                # uv-managed; deps split into base / teachers / tpu / tracking / dev
```

## Prerequisites

- Python ≥ 3.10, [`uv`](https://github.com/astral-sh/uv) for dependency management.
- A `.env` at the repo root with at least `HF_TOKEN=…`, `WANDB_API_KEY=…`, and (for TPU provisioning) `SDM_GCP_PROJECT=…`. See `sdm/dotenv.py` and `.tpu/_lib.sh` for the variables that get read. The `.env` itself is gitignored; the provisioning scripts source it for secrets.
- For TPU runs: a GCS bucket reachable from the slice (configs default to `gs://sdm-ckpts/<experiment>/`) and a baked TPU image with matching `torch` + `torch_xla` wheels.

```bash
uv sync --extra dev --extra tracking
```

The `teachers` extra installs heavyweight feature extractors (`ttsds`, `pyworld`, `wespeaker-unofficial`, FunASR, …) and is only needed on a host that runs the teachers offline — not on TPU training hosts where the configured teacher loads at startup.

## Running a finetune

**Local CPU smoke** (uses the tiny in-tree backbone — fast, no GPU/TPU needed):

```bash
uv run python -m sdm.train.run_distill --config configs/finetune_tiny.yaml
```

**Single-host GPU/CPU run of the reference config**:

```bash
uv run python -m sdm.train.run_distill --config configs/finetune_xlsr.yaml
```

**TPU (spot, with retry on preemption) — single experiment**:

```bash
./.tpu/provision_pitch.sh   # or provision_<any>.sh
```

This polls `gcloud tpu-vm create` until v6e-8 capacity opens in `europe-west4-a`, SCPs `.env` onto the VM, clones the repo, and launches `scripts/run_finetune.sh CONFIG=…` inside `tmux -s sdm`. `scripts/run_finetune.sh` handles SIGTERM on preemption (flush + exit 130) and loops `python -m sdm.train.run_distill` so `resume_from_latest` in the YAML picks up where we left off.

**TPU — fan out all finetunes**:

```bash
./.tpu/provision_all.sh
tail -f .tpu/logs/*.log
```

**Tear down everything**:

```bash
./.tpu/teardown.sh
```

## Running the probe matrix (Phase 2)

```bash
./.tpu/provision_probes.sh
```

Brings up a single `sdm-probes` v6e-8, runs `scripts/run_probes.sh` on the VM:
1. `consolidate_weights.py` pulls all 9 `final.pt` from `gs://sdm-ckpts/`,
2. `run_linear_probes.py` loads each backbone once, computes features +
   teacher targets on **100 utterances of `mythicinfinity/libritts dev.clean`**
   (override via `SDM_PROBE_DATASET` / `SDM_PROBE_SPLIT` / `SDM_PROBE_UTTERANCES`),
   then fits Ridge probes across the 9 × 9 grid (with `--layer-sweep` for the
   per-layer breakdown).
3. The resulting `matrix.json` is uploaded to `gs://sdm-ckpts/probes/<run-id>/`.

LibriTTS is the default because it is small, English, and well-known —
enough signal to sanity-check the factor groupings before scaling the probe
set up to a multilingual held-out slice.

## Tests

```bash
uv run pytest -q
```

Exercises the live path: streaming Emilia → chunk collate → XLS-R + configured teacher → masked loss → checkpoint I/O.

## Adding a new Teacher

See [`ARCHITECTURE.md#adding-a-teacher`](ARCHITECTURE.md#adding-a-teacher) for the full contract. Sketch:

1. Create `sdm/data/teachers/<kind>.py` exposing a `Teacher`-shaped class with `__call__(audio, chunk_mask, **ctx) -> Tensor[B, N_chunks, D]`.
2. Register the `<kind>` string in `sdm/data/teachers/__init__.py`.
3. Drop a `configs/finetune_<kind>.yaml` matching the field shape of the existing configs.
4. Copy a `.tpu/provision_<other>.sh` to `.tpu/provision_<kind>.sh`, change `NAME` and `CONFIG`, add it to `provision_all.sh` + `teardown.sh`.
5. `./.tpu/provision_<kind>.sh`.
