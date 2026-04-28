#!/usr/bin/env bash
# In-VM driver for the cross-teacher probe analysis (Phase 2).
#
# Runs after .tpu/provision_probes.sh has SCP'd .env onto the host. Pulls
# every finetune's final.pt from gs://sdm-ckpts, fits the probe matrix, then
# uploads the matrix back to gs://sdm-ckpts/probes/<run-id>/.

export EXPERIMENT='sdm-probes'
export CONFIG='configs/finetune_xlsr.yaml'   # used by _tpu_common.sh sanity check

# shellcheck disable=SC1091
source "$(dirname "$0")/_tpu_common.sh"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="runs/probes/${RUN_ID}"
CKPT_DIR="checkpoints/consolidated"
GCS_DEST="gs://sdm-ckpts/probes/${RUN_ID}"

EXPERIMENTS=(
    sdm-xlsr            # mHuBERT-147 generic
    sdm-dvector         # speaker
    sdm-pitch           # prosody — F0
    sdm-mpm             # prosody — MPM L7
    sdm-speaking-rate   # prosody — syllables/sec
    sdm-w2v2-asr        # intelligibility — wav2vec2 ASR encoder
    sdm-mwhisper        # intelligibility — Whisper-small encoder
    sdm-emotion2vec     # emotion (outside paper, included downstream)
    # sdm-wespeaker       # speaker
)

echo "[$EXPERIMENT] consolidating ${#EXPERIMENTS[@]} checkpoints into $CKPT_DIR"
uv run python scripts/consolidate_weights.py \
    --experiments "${EXPERIMENTS[@]}" \
    --out "$CKPT_DIR"

echo "[$EXPERIMENT] running probe matrix -> $OUT_DIR"
uv run python scripts/run_linear_probes.py \
    --consolidated "$CKPT_DIR" \
    --dataset "${SDM_PROBE_DATASET:-mythicinfinity/libritts}" \
    --split "${SDM_PROBE_SPLIT:-dev.clean}" \
    --probe-utterances "${SDM_PROBE_UTTERANCES:-100}" \
    --batch-size "${SDM_PROBE_BATCH_SIZE:-4}" \
    --layer-sweep \
    --out "$OUT_DIR" \
    ${WANDB_API_KEY:+--wandb}

echo "[$EXPERIMENT] uploading matrix to $GCS_DEST"
gsutil -q cp "$OUT_DIR/matrix.json" "$GCS_DEST/matrix.json"
gsutil -q cp -r "$CKPT_DIR/manifest.json" "$GCS_DEST/manifest.json"

echo "[$EXPERIMENT] done; matrix at $GCS_DEST/matrix.json"
