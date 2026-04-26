#!/usr/bin/env bash
# TPU sdm-allosaurus: distill Allosaurus phones-per-second (multilingual).
export EXPERIMENT="sdm-allosaurus"
export CONFIG="configs/finetune_allosaurus.yaml"

# shellcheck disable=SC1091
source "$(dirname "$0")/_tpu_common.sh"

# Allosaurus is CPU-only; targets are pre-extracted from a sampled subset
# (see scripts/extract_offline_targets.sh) and read from gs:// inside the loop.
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
