#!/usr/bin/env bash
# Create 8 preemptible (spot) v6e-8 TPU VMs, one per experiment.
#
# Prereqs:
#   - gcloud auth login + application-default login
#   - GCP project set in the environment ($SDM_GCP_PROJECT) or .env
#   - TRC quota for v6e-8 spot in the chosen zone
#
# Usage:
#   ./scripts/request_tpus.sh                     # create all 8
#   ./scripts/request_tpus.sh sdm-xlsr            # create just one
#   ./scripts/request_tpus.sh --delete            # delete all 8
#   ./scripts/request_tpus.sh --status            # check VM status
#
# Names chosen to match `configs/finetune_<exp>.yaml` and
# `scripts/run_<exp>.sh`, so the SSH user can find their script trivially.

set -uo pipefail

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

PROJECT="${SDM_GCP_PROJECT:?set SDM_GCP_PROJECT in .env}"
ZONE="${SDM_TPU_ZONE:-europe-west4-a}"      # known-good v6e pool used for smoke tests
ACCEL="${SDM_TPU_ACCEL:-v6e-8}"
RUNTIME="${SDM_TPU_RUNTIME:-v2-alpha-tpuv6e}"

EXPERIMENTS=(
    sdm-xlsr           # XLSR-53 generic, layer 8 (sequence)
    sdm-dvector        # d-Vector windowed
    sdm-wespeaker      # WeSpeaker windowed
    sdm-pitch          # WORLD F0 per-chunk mean
    sdm-mpm            # MPM layer 7 (English-biased; accepted)
    sdm-allosaurus     # Allosaurus speaking-rate per chunk
    sdm-w2v2-asr       # wav2vec2-large-xlsr-53 ASR encoder
    sdm-mwhisper       # Whisper-small multilingual encoder
)

create_one() {
    local name="$1"
    echo "[create] $name"
    gcloud compute tpus tpu-vm create "$name" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --accelerator-type="$ACCEL" \
        --version="$RUNTIME" \
        --spot
}

delete_one() {
    local name="$1"
    echo "[delete] $name"
    gcloud compute tpus tpu-vm delete "$name" \
        --project="$PROJECT" --zone="$ZONE" --force --quiet || true
}

status() {
    gcloud compute tpus tpu-vm list \
        --project="$PROJECT" --zone="$ZONE" \
        --filter="name~^sdm-"
}

case "${1:-}" in
    --delete)
        for n in "${EXPERIMENTS[@]}"; do delete_one "$n"; done
        ;;
    --status)
        status
        ;;
    "")
        for n in "${EXPERIMENTS[@]}"; do create_one "$n"; done
        echo
        echo "Run --status to watch provisioning. Once each is READY, ssh with:"
        echo "  gcloud compute tpus tpu-vm ssh <name> --zone=$ZONE --project=$PROJECT"
        ;;
    sdm-*)
        create_one "$1"
        ;;
    *)
        echo "Unknown arg: $1" >&2; exit 2
        ;;
esac
