#!/usr/bin/env bash
# Canonical TPU launcher for any sdm distillation finetune.
#
# Usage:
#   CONFIG=configs/finetune_xlsr_fairseq.yaml ./scripts/run_finetune.sh
#
# Sources scripts/_tpu_common.sh (clones/pulls repo, syncs deps, .env, env vars),
# then runs `sdm.train.run_distill --config $CONFIG` with a retry loop. SIGTERM
# on spot preemption: run_distill flushes a checkpoint and exits 130; the loop
# below relaunches and resume_from_latest in the YAML picks up where we left off.

: "${CONFIG:?must export CONFIG=configs/finetune_<teacher>.yaml}"

# Derive EXPERIMENT from the YAML filename (configs/finetune_xlsr_fairseq.yaml -> sdm-xlsr-fairseq).
config_basename=$(basename "$CONFIG" .yaml)
export EXPERIMENT="${EXPERIMENT:-sdm-${config_basename#finetune_}}"
export CONFIG

# shellcheck disable=SC1091
source "$(dirname "$0")/_tpu_common.sh"

attempt=0
MAX_RETRIES="${SDM_MAX_RETRIES:-50}"
RETRY_SLEEP="${SDM_RETRY_SLEEP:-30}"
until (( attempt >= MAX_RETRIES )); do
    attempt=$((attempt + 1))
    echo "[$EXPERIMENT] attempt $attempt"
    if uv run python -m sdm.train.run_distill --config "$CONFIG"; then
        exit 0
    fi
    rc=$?
    if (( rc == 130 )); then
        echo "[$EXPERIMENT] preempted (rc=$rc); retrying in ${RETRY_SLEEP}s"
    else
        echo "[$EXPERIMENT] crash (rc=$rc); retrying in ${RETRY_SLEEP}s"
    fi
    sleep "$RETRY_SLEEP"
done
echo "[$EXPERIMENT] gave up after $MAX_RETRIES" >&2
exit 1
