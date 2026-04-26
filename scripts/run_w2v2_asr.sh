#!/usr/bin/env bash
# TPU sdm-w2v2-asr: distill wav2vec2-large-xlsr-53 ASR encoder, per-chunk mean.
export EXPERIMENT="sdm-w2v2-asr"
export CONFIG="configs/finetune_w2v2_asr.yaml"

# shellcheck disable=SC1091
source "$(dirname "$0")/_tpu_common.sh"

attempt=0
MAX_RETRIES="${SDM_MAX_RETRIES:-50}"
until (( attempt >= MAX_RETRIES )); do
    attempt=$((attempt + 1))
    echo "[$EXPERIMENT] attempt $attempt"
    uv run python -m sdm.train.run_distill --config "$CONFIG" && exit 0
    sleep 30
done
echo "[$EXPERIMENT] gave up after $MAX_RETRIES" >&2
exit 1
