#!/usr/bin/env bash
# Local-GPU driver for the cross-teacher probe analysis (Phase 2).
#
# Mirrors scripts/run_probes.sh, but runs on a local CUDA box instead of a
# TPU. No torch_xla, no automatic GCS upload (artefacts go to runs/probes/
# locally; you can rsync them later if you want).
#
# Prereqs (one-time):
#   uv pip install --index-url https://download.pytorch.org/whl/cu128 \
#       'torch~=2.7.0' 'torchaudio~=2.7.0'
#   uv pip install pandas phonemizer pyworld funasr scikit-learn matplotlib \
#       wandb 'masked_prosody_model @ git+https://github.com/MiniXC/masked_prosody_model'
#   # plus host-side espeak-ng (or stage libs into .local-libs/, see flatpak
#   # bootstrap notes).
#
# Optional env vars:
#   SDM_PROBE_DEVICE=cuda    (default cuda)
#   SDM_PROBE_BATCH_SIZE=8   (default 8 — bigger than the TPU CPU run)
#   SDM_PROBE_DATASET, SDM_PROBE_SPLIT, SDM_PROBE_UTTERANCES — same as TPU
#   PHONEMIZER_ESPEAK_LIBRARY, ESPEAK_DATA_PATH, LD_LIBRARY_PATH —
#       set these if espeak isn't on the system path (e.g. flatpak sandbox).

set -uo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="runs/probes/${RUN_ID}"
CKPT_DIR="checkpoints/consolidated"

EXPERIMENTS=(
    sdm-xlsr-fairseq    # XLS-R 300M generic backbone (fairseq-init)
    sdm-dvector         # speaker
    sdm-pitch           # prosody — F0
    sdm-mpm             # prosody — MPM L7
    sdm-speaking-rate   # prosody — syllables/sec
    sdm-w2v2-asr        # intelligibility — wav2vec2 ASR encoder
    sdm-mwhisper        # intelligibility — Whisper-small encoder
    sdm-emotion2vec     # emotion (outside paper, included downstream)
    # sdm-wespeaker     # excluded; see ROADMAP.md
)

echo "[local] consolidating ${#EXPERIMENTS[@]} checkpoints into $CKPT_DIR"
.venv/bin/python scripts/consolidate_weights.py \
    --experiments "${EXPERIMENTS[@]}" \
    --out "$CKPT_DIR"

echo "[local] running probe matrix -> $OUT_DIR"
.venv/bin/python scripts/run_linear_probes.py \
    --consolidated "$CKPT_DIR" \
    --dataset "${SDM_PROBE_DATASET:-mythicinfinity/libritts}" \
    --split "${SDM_PROBE_SPLIT:-dev.clean}" \
    --probe-utterances "${SDM_PROBE_UTTERANCES:-500}" \
    --batch-size "${SDM_PROBE_BATCH_SIZE:-8}" \
    --device "${SDM_PROBE_DEVICE:-cuda}" \
    --layer-sweep \
    --n-jobs "${SDM_PROBE_N_JOBS:-16}" \
    --out "$OUT_DIR" \
    ${WANDB_API_KEY:+--wandb}

echo "[local] running analysis -> $OUT_DIR"
.venv/bin/python scripts/analyze_probes.py \
    --matrix "$OUT_DIR/matrix.json" \
    --out "$OUT_DIR" \
    ${WANDB_API_KEY:+--wandb} \
    --wandb-project sdm

echo "[local] done; artefacts at $OUT_DIR"
