#!/usr/bin/env bash
# Provision a small TPU VM for smoke testing. Defaults to v4-8 in
# us-central2-b (smallest on-demand v4 slice, single host, no spot churn).
#
# Usage:
#   ./scripts/tpu/provision_test_vm.sh [NAME] [ACCEL] [ZONE]
#
# Examples:
#   ./scripts/tpu/provision_test_vm.sh                    # sdm-test, v4-8
#   ./scripts/tpu/provision_test_vm.sh sdm-test v6e-8 us-east1-d
#
# After it comes up, ssh in and run scripts/tpu/bake_vm.sh, then
# scripts/tpu/smoke_test.sh.
set -euo pipefail

NAME="${1:-sdm-test}"
ACCEL="${2:-v4-8}"
ZONE="${3:-us-central2-b}"
PROJECT="${SDM_GCP_PROJECT:-ml-edinburgh}"

case "$ACCEL" in
    v4-*) RUNTIME="tpu-ubuntu2204-base" ;;
    v5e-*|v5litepod-*) RUNTIME="v2-alpha-tpuv5-lite" ;;
    v6e-*) RUNTIME="v2-alpha-tpuv6e" ;;
    *) echo "Unknown accelerator $ACCEL" >&2; exit 2 ;;
esac

echo "[provision] $NAME ($ACCEL, $RUNTIME) in $ZONE / $PROJECT"
gcloud compute tpus tpu-vm create "$NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --accelerator-type="$ACCEL" \
    --version="$RUNTIME"

echo
echo "[provision] up. SSH in with:"
echo "  gcloud compute tpus tpu-vm ssh $NAME --project=$PROJECT --zone=$ZONE --worker=all"
echo
echo "[provision] then on the VM:"
echo "  git clone git@github.com:ttsds/sdm.git && cd sdm"
echo "  bash scripts/tpu/bake_vm.sh"
echo "  bash scripts/tpu/smoke_test.sh"
echo
echo "[provision] when done:"
echo "  gcloud compute tpus tpu-vm delete $NAME --project=$PROJECT --zone=$ZONE"
