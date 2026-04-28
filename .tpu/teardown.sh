#!/usr/bin/env bash
# Tear down the per-experiment TPU VMs (one-shot delete).
# Use this to clean up after a finetune is done or to recover from a bad state.
set -uo pipefail

# Pull SDM_GCP_PROJECT (and other env defaults) from the repo .env so this
# script lives next to the rest of .tpu/ without needing manual exports.
SDM_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
ENV_FILE="${SDM_ENV_FILE:-${SDM_REPO_ROOT}/.env}"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

PROJECT="${SDM_GCP_PROJECT:?set SDM_GCP_PROJECT in .env}"
ZONE="${SDM_TPU_ZONE:-europe-west4-a}"

NAMES=(
    sdm-dvector
    sdm-wespeaker
    sdm-pitch
    sdm-mpm
    sdm-speaking-rate
    sdm-w2v2-asr
    sdm-mwhisper
    sdm-emotion2vec
    sdm-probes
)

for n in "${NAMES[@]}"; do
    echo "[teardown] $n"
    gcloud compute tpus tpu-vm delete "$n" \
        --project="$PROJECT" --zone="$ZONE" --quiet || true
done

# Also kill any local poll loops still trying to (re)create.
pkill -f 'provision_.*\.sh' 2>/dev/null || true
echo "[teardown] done."
