#!/usr/bin/env bash
# Provision a TPU and launch finetune_mwhisper. Secrets come from repo .env.
set -uo pipefail

NAME='sdm-mwhisper'
CONFIG='configs/finetune_mwhisper.yaml'

# shellcheck disable=SC1091
source "$(dirname "$0")/_lib.sh"
provision
