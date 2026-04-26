# SDM — Status

Last updated: 2026-04-25 (post-pivot from `vocex` codec plan).

## Project in one paragraph

Train **8 separate finetunes of mHuBERT-147** on a streaming, language-shuffled
slice of `amphion/Emilia-Dataset` — each finetune distilling **one** TTSDS-style
teacher representation. All audio is pooled into **1-second chunks** (truncate
beyond 32 chunks). Use only **10% of the training stream** per run. Once all
8 finish, pull the backbones and run **chunk-level linear probes** to build an
8×8 cross-prediction matrix; the empirical clusters in that matrix replace
TTSDS's a-priori factor taxonomy. v6e-8 spot TPUs, one per experiment, W&B
for monitoring.

## The 8 experiments (locked)

| TPU name | Teacher | Output per chunk |
|---|---|---|
| `sdm-xlsr` | XLSR-53 (`facebook/wav2vec2-xls-r-300m`) layer 8 | `(1024,)` |
| `sdm-dvector` | d-Vector (ttsds package, torchscript) | `(256,)` |
| `sdm-wespeaker` | `pyannote/wespeaker-voxceleb-resnet34-LM` underlying ResNet | `(256,)` |
| `sdm-pitch` | WORLD F0 mean | `(1,)` |
| `sdm-mpm` | MPM L7 (`cdminix/masked_prosody_model`) — English bias accepted | `(256,)` |
| `sdm-allosaurus` | Allosaurus phones/sec (offline-extracted) | `(1,)` |
| `sdm-w2v2-asr` | `jonatasgrosman/wav2vec2-large-xlsr-53-english` encoder | `(1024,)` |
| `sdm-mwhisper` | `openai/whisper-small` encoder | `(768,)` |

**Deferred:** `emotion2vec` — count was 9, only 8 TPUs available. Easy to swap
in by replacing one of the speaker / prosody slots.

## What's done

| Area | Files | Notes |
|---|---|---|
| Repo layout, deps, uv | `pyproject.toml`, `.gitignore` | TPU-safe install path; heavyweight teacher deps isolated from TPU hosts |
| W&B + GCS + spot resume plumbing | `sdm/train/{wandb_utils,io,preempt}.py` | Master-only init, gs:// ckpt I/O, SIGTERM flush, outer retry loop |
| `.env` loader | `sdm/dotenv.py` | Sources `WANDB_API_KEY` etc. on TPU hosts |
| TPU request | `scripts/request_tpus.sh` | Direct `tpu-vm create --spot` for all 8 v6e-8 spots; `--status`, `--delete`, single-name modes |
| Per-TPU run scripts | `scripts/run_<exp>.sh` × 8 + `_tpu_common.sh` | Idempotent: clone+pull, install uv, install torch_xla matched to torch, source `.env`, retry loop |
| Per-experiment configs | `configs/finetune_<exp>.yaml` × 8 | mHuBERT backbone, 1s chunks, max 32 chunks, 10% fraction, gs:// ckpt, W&B run+tags |
| Streaming Emilia loader | `sdm/data/streaming_emilia.py` | Streams/shuffles Emilia, resamples to 16 kHz mono, chunks/pads to fixed `(max_chunks, samples)` tensors |
| mHuBERT distill model | `sdm/modeling/distill_model.py` | HF AutoModel wrapper, chunk pooling, projection head, `load_backbone()` for probes |
| Distillation entrypoint | `sdm/train/run_distill.py` | YAML config, streaming loader, teacher dispatch, FSDP-via-XLA loop, masked MSE, gs:// checkpoints |
| First teacher | `sdm/data/teachers/hf_ssl.py` | XLS-R / wav2vec2-style HF SSL teacher for `sdm-xlsr` |
| Weight consolidation | `scripts/consolidate_weights.py` | Pulls `final.pt` from each gs:// bucket, strips heads, writes manifest |
| Cross-prediction probe | `scripts/run_linear_probes.py` | Wired, but only usable for teacher kinds that have implementations |

## What's NOT yet implemented (next agent: start here)

`sdm-xlsr` is the first runnable post-pivot experiment. The remaining seven
configs still need their teacher modules before their run scripts can train:

1. **Remaining `sdm/data/teachers/` modules** — one module per `teacher.kind` referenced by the
  configs:
   - `hf_ctc.py` (w2v2-asr) — same shape, picks last-hidden mean per chunk.
   - `whisper_encoder.py` (mwhisper) — runs `model.model.encoder` on full
     utterance, mean-pools encoder frames per chunk. **Note:** deviates from
     TTSDS's 1500-d cross-layer/hidden-mean trick; called out in the configs.
   - `mpm.py` — wraps `MaskedProsodyModel`, accepts 22050 Hz reample.
   - `dvector_torchscript.py` — torchscript d-vector + Wav2Mel; sliding-window
     internally to 1s/chunk.
   - `wespeaker_resnet34.py` — strips the pyannote.audio Inference wrapper,
     loads the underlying ResNet so it can run on TPU.
   - `pyworld_f0.py` — CPU-only; computes per-chunk pitch mean.
   - `allosaurus_offline.py` — reads pre-extracted shards from gs://; needs an
     offline extractor (`scripts/extract_offline_targets.py`, also TBD).
   - All teachers expose `__call__(audio: Tensor, chunk_seconds: float) ->
     Tensor of shape (B, N_chunks, D)`.

2. **`scripts/extract_offline_targets.py`** — CPU/GPU box job for `allosaurus`
   (and any other non-TPU teacher) that streams Emilia, computes per-chunk
   targets, writes sharded `.npz` to `gs://sdm-cache/teacher_cache/<teacher>/`.

3. **Tests** — add one output-shape test for each remaining teacher on a 5s
  random waveform (gated behind `pytest.importorskip` for heavy deps).

## How to run end-to-end (once 1-5 above land)

```bash
# 0. one-time prereqs locally
cp .env.example .env  # fill in WANDB_API_KEY, SDM_GCP_PROJECT, SDM_TPU_ZONE
gcloud auth login && gcloud auth application-default login

# 1. request all 8 spot TPU VMs
./scripts/request_tpus.sh
./scripts/request_tpus.sh --status   # poll until READY

# 2. per TPU: scp .env once, then ssh + run
for n in sdm-xlsr sdm-dvector sdm-wespeaker sdm-pitch sdm-mpm \
         sdm-allosaurus sdm-w2v2-asr sdm-mwhisper; do
    gcloud compute tpus tpu-vm scp .env "$n":~/sdm/.env --zone=europe-west4-a
done
# then ssh to each and `bash ~/sdm/scripts/run_<exp>.sh`

# 3. once all 8 final.pt's are in gs://sdm-ckpts/<exp>/
uv run python scripts/consolidate_weights.py \
    --experiments sdm-xlsr sdm-dvector sdm-wespeaker sdm-pitch \
                  sdm-mpm sdm-allosaurus sdm-w2v2-asr sdm-mwhisper \
    --out checkpoints/consolidated

# 4. cross-prediction probe matrix
uv run python scripts/run_linear_probes.py \
    --consolidated checkpoints/consolidated \
    --probe-utterances 1000 \
    --layer-sweep \
    --out runs/probes/v0 \
    --wandb
```

## Open product/design questions (still unanswered by user)

- **A. Probe layer scope.** Single fixed layer per SDM_i, or sweep all 12
  encoder layers? `--layer-sweep` flag is wired in `run_linear_probes.py`
  for either choice.
- **B. emotion2vec target.** If we re-add emotion2vec as a 9th experiment:
  utterance-pooled features, frame-level features, or 9-class emotion logits?
- **C. Backbone training mode.** Configs assume **full-finetune** of mHuBERT-
  147 (necessary for the cross-probe — frozen backbones would all share the
  same latent space). Confirm this is intended; LoRA/adapter would also work
  if the user wants smaller per-experiment ckpts.

## Reference

- Plan file (original; partially superseded): `~/.claude/plans/this-is-a-new-curried-badger.md`
- Upstream TTSDS source (cloned): `/tmp/ttsds`
- Upstream ml-tidbits (wristband loss, deferred ablation): `/tmp/ml-tidbits`
- Memory dir: `~/.claude/projects/-home-cminixhofer-Documents-Repositories-vocex/memory/`
