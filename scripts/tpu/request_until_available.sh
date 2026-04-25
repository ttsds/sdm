#!/usr/bin/env bash
# Wait-for-capacity TPU provisioning. Two strategies:
#
#   --queued (default): use `gcloud compute tpus queued-resources create`,
#       which holds the request server-side until capacity opens. Best for
#       TRC, no client polling needed. Cancel with Ctrl-C and:
#         gcloud compute tpus queued-resources delete <NAME> --zone=<ZONE>
#
#   --poll: client-side retry loop calling `tpu-vm create`. Useful when
#       queued-resources is restricted (some TRC zones / older runtimes).
#
# Usage:
#   ./scripts/tpu/request_until_available.sh [NAME] [ACCEL] [ZONE] [--poll|--queued]
#
# Examples:
#   ./scripts/tpu/request_until_available.sh                     # sdm-test, v4-8, us-central2-b, queued
#   ./scripts/tpu/request_until_available.sh sdm-test v6e-8 us-east1-d --poll
set -uo pipefail

NAME="${1:-sdm-test}"
ACCEL="${2:-v4-8}"
ZONE="${3:-us-central2-b}"
MODE="${4:---queued}"
PROJECT="${SDM_GCP_PROJECT:-ml-edinburgh}"
SLEEP="${SDM_PROVISION_SLEEP:-60}"

case "$ACCEL" in
    v4-*) RUNTIME="tpu-ubuntu2204-base" ;;
    v5e-*|v5litepod-*) RUNTIME="v2-alpha-tpuv5-lite" ;;
    v6e-*) RUNTIME="v2-alpha-tpuv6e" ;;
    *) echo "Unknown accelerator $ACCEL" >&2; exit 2 ;;
esac

case "$MODE" in
  --queued)
    echo "[queued] requesting $NAME ($ACCEL, $RUNTIME) in $ZONE / $PROJECT"
    echo "[queued] gcloud will block until capacity opens; Ctrl-C to cancel."
    set -e
    gcloud compute tpus queued-resources create "$NAME" \
        --node-id="$NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --accelerator-type="$ACCEL" \
        --runtime-version="$RUNTIME"
    echo "[queued] queued. Watch with:"
    echo "  gcloud compute tpus queued-resources list --project=$PROJECT --zone=$ZONE"
    echo "[queued] when ACTIVE, ssh in:"
    echo "  gcloud compute tpus tpu-vm ssh $NAME --project=$PROJECT --zone=$ZONE --worker=all"
    ;;
  --poll)
    echo "[poll] retrying $NAME ($ACCEL, $RUNTIME) in $ZONE every ${SLEEP}s"
    attempt=0
    while true; do
        attempt=$((attempt + 1))
        ts="$(date -Iseconds)"
        echo "[poll] $ts attempt $attempt"
        if gcloud compute tpus tpu-vm create "$NAME" \
                --project="$PROJECT" \
                --zone="$ZONE" \
                --accelerator-type="$ACCEL" \
                --version="$RUNTIME" 2>/tmp/sdm_provision_err; then
            echo "[poll] up after $attempt attempts."
            echo "  gcloud compute tpus tpu-vm ssh $NAME --project=$PROJECT --zone=$ZONE --worker=all"
            exit 0
        fi
        err="$(cat /tmp/sdm_provision_err)"
        echo "$err" | tail -5
        # If it's a hard error (not capacity / not service availability), bail.
        if echo "$err" | grep -qE 'no more capacity|RESOURCE_EXHAUSTED|UNAVAILABLE|resourceExhausted|Stockout|currently unavailable'; then
            echo "[poll] capacity stockout; sleeping ${SLEEP}s"
        elif echo "$err" | grep -qE 'tenant project creation'; then
            echo "[poll] tenant-project creation hiccup; sleeping ${SLEEP}s"
        else
            echo "[poll] non-retryable error; aborting." >&2
            exit 1
        fi
        sleep "$SLEEP"
    done
    ;;
  *)
    echo "Unknown mode $MODE (use --queued or --poll)" >&2; exit 2
    ;;
esac
