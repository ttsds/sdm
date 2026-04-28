#!/usr/bin/env bash
# Provision a TPU and launch finetune_w2v2_asr. Secrets come from repo .env.
set -uo pipefail

NAME='sdm-w2v2-asr'
CONFIG='configs/finetune_w2v2_asr.yaml'

# shellcheck disable=SC1091
source "$(dirname "$0")/_lib.sh"
provision
