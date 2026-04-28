#!/usr/bin/env bash
# Provision a TPU and launch finetune_emotion2vec. Secrets come from repo .env.
set -uo pipefail

NAME='sdm-emotion2vec'
CONFIG='configs/finetune_emotion2vec.yaml'

# shellcheck disable=SC1091
source "$(dirname "$0")/_lib.sh"
provision
