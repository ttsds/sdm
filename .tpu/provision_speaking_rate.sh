#!/usr/bin/env bash
# Provision a TPU and launch finetune_speaking_rate. Secrets come from repo .env.
set -uo pipefail

NAME='sdm-speaking-rate'
CONFIG='configs/finetune_speaking_rate.yaml'

# shellcheck disable=SC1091
source "$(dirname "$0")/_lib.sh"
provision
