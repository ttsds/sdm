#!/usr/bin/env bash
# Provision a TPU and launch finetune_pitch. Secrets come from repo .env.
set -uo pipefail

NAME='sdm-pitch'
CONFIG='configs/finetune_pitch.yaml'

# shellcheck disable=SC1091
source "$(dirname "$0")/_lib.sh"
provision
