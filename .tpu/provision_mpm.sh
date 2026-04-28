#!/usr/bin/env bash
# Provision a TPU and launch finetune_mpm. Secrets come from repo .env.
set -uo pipefail

NAME='sdm-mpm'
CONFIG='configs/finetune_mpm.yaml'

# shellcheck disable=SC1091
source "$(dirname "$0")/_lib.sh"
provision
