#!/usr/bin/env bash
# Provision a TPU VM and run the cross-teacher probe analysis (Phase 2).
#
# The probe driver (scripts/run_probes.sh) runs on the TPU host:
#   1. consolidate_weights.py pulls each finetune's final.pt from gs://sdm-ckpts/
#   2. run_linear_probes.py fits Ridge / MLP probes from each backbone's
#      hidden states onto every other teacher's chunk targets and writes an
#      R^2 matrix.
#   3. The matrix and per-pair logs are uploaded to gs://sdm-ckpts/probes/<run>/.
#
# We reuse the same v6e-8 shape as the finetunes — the host CPU does most of
# the work for Ridge fitting; backbone forward passes use the TPU.
set -uo pipefail

NAME='sdm-probes'
LAUNCH_CMD='bash scripts/run_probes.sh'
SDM_REMOTE_LOG='~/sdm-probes.log'
export LAUNCH_CMD SDM_REMOTE_LOG

# shellcheck disable=SC1091
source "$(dirname "$0")/_lib.sh"
provision
