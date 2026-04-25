#!/usr/bin/env bash
# Launch sdm pretraining on a TPU pod via PyTorch/XLA.
#
# Prereqs (bake into the TPU VM image once per generation):
#   pip install -e '.[neucodec,tracking]'
#   pip install torch~=$(python -c 'import torch; print(torch.__version__.split("+")[0])') \
#               torch_xla -f https://storage.googleapis.com/libtpu-releases/index.html
#   gcloud auth application-default login   # for gs:// checkpoint I/O
#   wandb login
#
# Usage:
#   ./scripts/launch_tpu_pretrain.sh configs/pretrain_base.yaml
#
# Spot preemption: SIGTERM is trapped inside the loop; the process flushes a
# checkpoint to gs://sdm-ckpts/<run>/latest.pt and exits 130. The retry loop
# below relaunches; resume_from_latest in the config picks up where we left
# off.
set -uo pipefail

CONFIG="${1:-configs/pretrain_base.yaml}"
MAX_RETRIES="${SDM_MAX_RETRIES:-100}"
RETRY_SLEEP="${SDM_RETRY_SLEEP:-30}"

# Load secrets (WANDB_API_KEY, etc.) from a local .env if present. Variables
# already exported by the environment win.
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

export PJRT_DEVICE=TPU
export XLA_USE_BF16=1
export PT_XLA_DEBUG_LEVEL=0
# WANDB_API_KEY should be set in the TPU VM environment / instance metadata.

attempt=0
while (( attempt < MAX_RETRIES )); do
    attempt=$((attempt + 1))
    echo "[launch] attempt $attempt for $CONFIG"
    python -m sdm.train.run_pretrain --config "$CONFIG"
    rc=$?
    if (( rc == 0 )); then
        echo "[launch] training finished cleanly"
        exit 0
    fi
    if (( rc == 130 )); then
        echo "[launch] preempted (rc=$rc); retrying in ${RETRY_SLEEP}s"
    else
        echo "[launch] crash (rc=$rc); retrying in ${RETRY_SLEEP}s"
    fi
    sleep "$RETRY_SLEEP"
done

echo "[launch] giving up after $MAX_RETRIES attempts" >&2
exit 1
