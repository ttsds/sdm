#!/usr/bin/env bash
# Shared helpers for the .tpu/ provision scripts.
#
# Secrets (WANDB_API_KEY, HF_TOKEN, optional SDM_GCP_PROJECT) are read from
# the repo-root .env, which is gitignored. Each provision_*.sh just sets:
#   NAME       — TPU VM name
#   CONFIG     — configs/finetune_<x>.yaml on the VM (relative to repo root)
# (or, for non-finetune launches like provision_probes.sh, overrides
# LAUNCH_CMD with the bash command to run inside tmux on the VM)
# then sources this file and calls:
#   provision

set -uo pipefail

# Repo root = parent of .tpu/.
SDM_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
ENV_FILE="${SDM_ENV_FILE:-${SDM_REPO_ROOT}/.env}"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
else
    echo "[_lib] missing $ENV_FILE — copy .env.example and fill in WANDB_API_KEY / HF_TOKEN" >&2
    exit 2
fi

# Defaults (override per-script before sourcing if needed).
PROJECT="${SDM_GCP_PROJECT:-ml-edinburgh}"
ZONE="${SDM_TPU_ZONE:-europe-west4-a}"
ACCEL="${SDM_TPU_ACCEL:-v6e-8}"
RUNTIME="${SDM_TPU_RUNTIME:-v2-alpha-tpuv6e}"
SPOT_FLAG="${SDM_TPU_SPOT_FLAG:---spot}"
SLEEP="${SDM_PROVISION_SLEEP:-30}"
REPO_URL="${SDM_REPO_URL:-https://github.com/ttsds/sdm.git}"
SDM_BRANCH="${SDM_BRANCH:-main}"

LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
mkdir -p "$LOG_DIR"

RETRY_RE='no more capacity|Insufficient capacity|RESOURCE_EXHAUSTED|UNAVAILABLE|resourceExhausted|Stockout|currently unavailable|tenant project creation|"code": 8|"code": 10|HttpError|503|504|deadline exceeded|Internal error|already exists'

_log() { printf '[%s] [%s] %s\n' "$(date -Iseconds)" "${NAME:-?}" "$*"; }

poll_create() {
    local errfile="/tmp/sdm_provision_err.${NAME}"
    local attempt=0
    while true; do
        attempt=$((attempt + 1))
        _log "create attempt $attempt ($ACCEL $RUNTIME $ZONE $SPOT_FLAG)"
        if gcloud compute tpus tpu-vm describe "$NAME" \
                --project="$PROJECT" --zone="$ZONE" >/dev/null 2>&1; then
            _log "VM already exists"
            return 0
        fi
        if gcloud compute tpus tpu-vm create "$NAME" \
                --project="$PROJECT" \
                --zone="$ZONE" \
                --accelerator-type="$ACCEL" \
                --version="$RUNTIME" \
                $SPOT_FLAG 2>"$errfile"; then
            _log "created after $attempt attempts"
            return 0
        fi
        tail -3 "$errfile" 2>/dev/null | sed "s/^/[$NAME]   /"
        if grep -aqE "$RETRY_RE" "$errfile" 2>/dev/null; then
            _log "retryable; sleep ${SLEEP}s"
        else
            _log "unrecognised error; retrying anyway (Ctrl-C to abort) sleep ${SLEEP}s"
        fi
        sleep "$SLEEP"
    done
}

wait_ready() {
    local attempt=0
    while true; do
        attempt=$((attempt + 1))
        local state
        state=$(gcloud compute tpus tpu-vm describe "$NAME" \
            --project="$PROJECT" --zone="$ZONE" \
            --format='value(state)' 2>/dev/null || echo "UNKNOWN")
        _log "state=$state (poll $attempt)"
        if [[ "$state" == "READY" ]]; then return 0; fi
        if [[ "$state" == "PREEMPTED" || "$state" == "TERMINATED" || "$state" == "FAILED" ]]; then
            _log "bad state $state; recreating"
            return 1
        fi
        sleep 15
    done
}

# Build a .env file on the VM from secrets we already have on the laptop.
push_env() {
    local tmp; tmp=$(mktemp)
    cat >"$tmp" <<EOF
WANDB_API_KEY=${WANDB_API_KEY:?}
HF_TOKEN=${HF_TOKEN:?}
SDM_GCP_PROJECT=${PROJECT}
EOF
    _log "scp .env -> $NAME:~/.env.sdm"
    gcloud compute tpus tpu-vm scp "$tmp" "$NAME:~/.env.sdm" \
        --project="$PROJECT" --zone="$ZONE" --worker=all
    rm -f "$tmp"
}

# Clone the repo (if needed), drop .env into place, and launch the configured
# command inside tmux so it survives the SSH disconnect. Idempotent.
#
# Default LAUNCH_CMD runs run_finetune.sh against $CONFIG. Probe / eval scripts
# override LAUNCH_CMD before sourcing this file (and may leave CONFIG empty).
launch_run() {
    local launch_cmd="${LAUNCH_CMD:-CONFIG=${CONFIG} bash scripts/run_finetune.sh}"
    local log_path="${SDM_REMOTE_LOG:-~/sdm-run.log}"
    local remote_cmd
    remote_cmd=$(cat <<EOF
set -euo pipefail
if [[ ! -d \$HOME/sdm/.git ]]; then
    git clone --branch ${SDM_BRANCH} ${REPO_URL} \$HOME/sdm
fi
cd \$HOME/sdm
git fetch --quiet origin
git checkout ${SDM_BRANCH}
git pull --ff-only
cp \$HOME/.env.sdm \$HOME/sdm/.env
chmod 600 \$HOME/sdm/.env
sudo apt-get install -y tmux >/dev/null 2>&1 || true
if tmux has-session -t sdm 2>/dev/null; then
    echo "tmux session 'sdm' already running; not relaunching"
else
    tmux new-session -d -s sdm "${launch_cmd} 2>&1 | tee -a ${log_path}"
fi
tmux ls
EOF
    )
    _log "launch via tmux: ${launch_cmd}"
    gcloud compute tpus tpu-vm ssh "$NAME" \
        --project="$PROJECT" --zone="$ZONE" --worker=all \
        --command="$remote_cmd"
}

provision() {
    : "${NAME:?provision_*.sh must set NAME}"
    if [[ -z "${LAUNCH_CMD:-}" ]]; then
        : "${CONFIG:?provision_*.sh must set CONFIG (or override LAUNCH_CMD)}"
    fi
    : "${WANDB_API_KEY:?WANDB_API_KEY must be set in .env}"
    : "${HF_TOKEN:?HF_TOKEN must be set in .env}"

    while true; do
        poll_create
        if wait_ready; then break; fi
        # bad state — try delete + recreate
        gcloud compute tpus tpu-vm delete "$NAME" \
            --project="$PROJECT" --zone="$ZONE" --quiet || true
        sleep 10
    done
    push_env
    launch_run
    _log "DONE — tail ${SDM_REMOTE_LOG:-~/sdm-run.log} on $NAME for output"
}
