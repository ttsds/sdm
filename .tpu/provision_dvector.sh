#!/usr/bin/env bash
# Provision a TPU and launch finetune_dvector. Secrets come from repo .env.
set -uo pipefail

NAME='sdm-dvector'
CONFIG='configs/finetune_dvector.yaml'

# shellcheck disable=SC1091
source "$(dirname "$0")/_lib.sh"
provision
