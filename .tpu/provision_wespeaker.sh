#!/usr/bin/env bash
# Provision a TPU and launch finetune_wespeaker. Secrets come from repo .env.
set -uo pipefail

NAME='sdm-wespeaker'
CONFIG='configs/finetune_wespeaker.yaml'

# shellcheck disable=SC1091
source "$(dirname "$0")/_lib.sh"
provision
