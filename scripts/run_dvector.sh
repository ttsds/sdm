#!/usr/bin/env bash
# TPU sdm-dvector: distill d-Vector windowed speaker embeddings.
export EXPERIMENT="sdm-dvector"
export CONFIG="configs/finetune_dvector.yaml"

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
