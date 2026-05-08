#!/usr/bin/env bash
# Provision + launch the 9 wristband ablation TPUs.
#
# Layout (chosen by user; see TPU_RESOURCES.md for pool background):
#   - 8 x v6e-8 spot in europe-west4-a (sdm-finetune-headline pool)
#   - 1 x v5e-8 spot in us-central1-a   (sdm-finetune-extra pool)
#
# Naming follows the convention used by scripts/run_finetune.sh:
#   VM = sdm-<exp>-wristband, CONFIG = configs/finetune_<exp>_wristband.yaml
#
# Subcommands:
#   --create   queue all 9 resources (best-effort spot, returns once queued)
#   --status   show queued-resource + VM state for all 9
#   --launch   scp .env, scp this repo's tip ref, ssh and start training
#              in a tmux session named "sdm" on each VM that's ACTIVE
#   --delete   tear down all 9 queued resources
#   <name>     run the named subcommand for a single experiment

set -uo pipefail

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

PROJECT="${SDM_GCP_PROJECT:?set SDM_GCP_PROJECT in .env}"
V6E_ZONE="${SDM_TPU_ZONE_V6E:-europe-west4-a}"
V4_ZONE="${SDM_TPU_ZONE_V4:-us-central2-b}"
V6E_RUNTIME="v2-alpha-tpuv6e"
V4_RUNTIME="tpu-ubuntu2204-base"

# (vm_name, config, accel, zone, runtime)
ROWS=(
    "sdm-xlsr-fairseq-wristband|configs/finetune_xlsr_fairseq_wristband.yaml|v6e-8|$V6E_ZONE|$V6E_RUNTIME"
    "sdm-dvector-wristband|configs/finetune_dvector_wristband.yaml|v6e-8|$V6E_ZONE|$V6E_RUNTIME"
    "sdm-pitch-wristband|configs/finetune_pitch_wristband.yaml|v6e-8|$V6E_ZONE|$V6E_RUNTIME"
    "sdm-mpm-wristband|configs/finetune_mpm_wristband.yaml|v6e-8|$V6E_ZONE|$V6E_RUNTIME"
    "sdm-mwhisper-wristband|configs/finetune_mwhisper_wristband.yaml|v6e-8|$V6E_ZONE|$V6E_RUNTIME"
    "sdm-w2v2-asr-wristband|configs/finetune_w2v2_asr_wristband.yaml|v6e-8|$V6E_ZONE|$V6E_RUNTIME"
    "sdm-emotion2vec-wristband|configs/finetune_emotion2vec_wristband.yaml|v6e-8|$V6E_ZONE|$V6E_RUNTIME"
    "sdm-speaking-rate-wristband|configs/finetune_speaking_rate_wristband.yaml|v6e-8|$V6E_ZONE|$V6E_RUNTIME"
    "sdm-xlsr-wristband|configs/finetune_xlsr_wristband.yaml|v4-8|$V4_ZONE|$V4_RUNTIME"
)

row_for() {
    local name="$1"
    for r in "${ROWS[@]}"; do
        IFS='|' read -r n _ _ _ _ <<<"$r"
        if [[ "$n" == "$name" ]]; then echo "$r"; return 0; fi
    done
    return 1
}

create_one() {
    local row="$1"
    IFS='|' read -r name cfg accel zone runtime <<<"$row"
    # Skip if VM already exists.
    if gcloud compute tpus tpu-vm describe "$name" --project="$PROJECT" --zone="$zone" \
            --format="value(name)" >/dev/null 2>&1; then
        echo "[create] $name: already exists, skipping"
        return 0
    fi
    mkdir -p .tpu-logs
    local log=".tpu-logs/${name}.create.log"
    local err=".tpu-logs/${name}.create.err"
    # Retry-until-stockout-clears loop. Each `tpu-vm create --spot` call blocks
    # ~15 min before failing on RESOURCE_EXHAUSTED; we just keep retrying. The
    # caller backgrounds this so all VMs poll in parallel.
    local RETRY_RE='no more capacity|Insufficient capacity|RESOURCE_EXHAUSTED|UNAVAILABLE|resourceExhausted|Stockout|currently unavailable|"code": 8|"code": 10|HttpError|503|504|deadline exceeded|Internal error'
    local attempt=0
    while true; do
        attempt=$((attempt + 1))
        echo "[$(date -Iseconds)] [$name] attempt $attempt" | tee -a "$log"
        if gcloud compute tpus tpu-vm create "$name" \
                --project="$PROJECT" \
                --zone="$zone" \
                --accelerator-type="$accel" \
                --version="$runtime" \
                --spot >>"$log" 2>"$err"; then
            echo "[$(date -Iseconds)] [$name] CREATED after $attempt attempts" | tee -a "$log"
            return 0
        fi
        tail -3 "$err" | tee -a "$log" >/dev/null
        if grep -aqE "$RETRY_RE" "$err" 2>/dev/null; then
            echo "[$name] retryable error, retrying immediately" >> "$log"
        else
            echo "[$name] unknown error (retrying anyway):" >> "$log"
            cat "$err" >> "$log"
        fi
    done
}

delete_one() {
    local row="$1"
    IFS='|' read -r name _ _ zone _ <<<"$row"
    echo "[delete] $name (zone=$zone)"
    gcloud compute tpus tpu-vm delete "$name" \
        --project="$PROJECT" --zone="$zone" --quiet 2>/dev/null || true
}

status_all() {
    declare -A ZONES
    for r in "${ROWS[@]}"; do
        IFS='|' read -r name _ _ zone _ <<<"$r"
        ZONES["$zone"]=1
    done
    for z in "${!ZONES[@]}"; do
        echo "=== $z ==="
        gcloud compute tpus tpu-vm list \
            --project="$PROJECT" --zone="$z" \
            --filter="name~-wristband$" \
            --format="table(name,state,acceleratorType)" 2>/dev/null
    done
}

launch_one() {
    local row="$1"
    IFS='|' read -r name cfg _ zone _ <<<"$row"
    local ssh="gcloud compute tpus tpu-vm ssh $name --project=$PROJECT --zone=$zone --worker=all"
    local scp="gcloud compute tpus tpu-vm scp --project=$PROJECT --zone=$zone --worker=all"
    mkdir -p .tpu-logs
    local log=".tpu-logs/${name}.launch.log"

    echo "[launch] $name -> $cfg" | tee -a "$log"
    # Wait up to 12h for VM to be READY (covers spot-stockout retry loops).
    for i in $(seq 1 1440); do
        local state
        state=$(gcloud compute tpus tpu-vm describe "$name" \
                --project="$PROJECT" --zone="$zone" \
                --format="value(state)" 2>/dev/null)
        if [[ "$state" == "READY" ]]; then break; fi
        if (( i % 10 == 0 )); then
            echo "[launch] $name state=$state (attempt $i); sleeping 30s" >> "$log"
        fi
        sleep 30
    done

    # Clone or fast-forward sdm to origin/main.
    $ssh --command "test -d ~/sdm/.git || git clone https://github.com/ttsds/sdm.git ~/sdm; cd ~/sdm && git fetch --quiet origin && git reset --hard origin/main" >> "$log" 2>&1

    # Upload .env (WANDB_API_KEY, HF_TOKEN, SDM_GCP_PROJECT).
    $scp .env "$name:~/sdm/.env" >> "$log" 2>&1

    # Idempotent launch: only start training if no `sdm` tmux session
    # exists. Otherwise leave the running one alone — re-running this
    # function on a VM that's already training would otherwise kill its
    # tmux and restart from the latest checkpoint (or step 0 if none),
    # wasting compute.
    $ssh --command "sudo apt-get install -y tmux >/dev/null 2>&1 || true; if tmux has-session -t sdm 2>/dev/null; then echo '[launch] tmux sdm already running, skipping'; else cd ~/sdm && tmux new -d -s sdm \"CONFIG=$cfg ./scripts/run_finetune.sh > train.log 2>&1\" && echo '[launch] tmux sdm started'; fi" >> "$log" 2>&1
    echo "[launch] $name: done" | tee -a "$log"
}

# Detached: poll-create-then-launch for one row. This is what
# --create-and-launch-detached spawns per VM.
create_and_launch_one() {
    local row="$1"
    create_one "$row" || return $?
    launch_one "$row"
}

case "${1:-}" in
    --create)
        # Foreground (parallel, blocks): all 9 retry until success.
        for r in "${ROWS[@]}"; do create_one "$r" & done; wait
        echo
        echo "All 9 created. Run: $0 --launch"
        ;;
    --create-detached)
        # Background detached: each retry loop runs as nohup, logs to
        # .tpu-logs/<name>.create.log. Survives shell exit. Check progress
        # with: $0 --status   or   tail -f .tpu-logs/*.create.log
        mkdir -p .tpu-logs
        for r in "${ROWS[@]}"; do
            IFS='|' read -r name _ _ _ _ <<<"$r"
            pidfile=".tpu-logs/${name}.create.pid"
            if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
                echo "[detach] $name: already polling (pid $(cat "$pidfile"))"
                continue
            fi
            nohup bash -c "
                $(declare -f create_one)
                PROJECT='$PROJECT'
                create_one '$r'
            " >/dev/null 2>&1 &
            echo $! > "$pidfile"
            echo "[detach] $name: pid $(cat "$pidfile") (log .tpu-logs/${name}.create.log)"
        done
        echo
        echo "All retry loops detached. Check with:"
        echo "  $0 --status"
        echo "  ls .tpu-logs/ ; tail -f .tpu-logs/<name>.create.log"
        ;;
    --all-detached)
        # Background detached: each VM runs poll-create-then-launch in one
        # nohup process. As soon as a VM goes READY, training starts. Check
        # with: $0 --status   or   tail -f .tpu-logs/<name>.{create,launch}.log
        mkdir -p .tpu-logs
        for r in "${ROWS[@]}"; do
            IFS='|' read -r name _ _ _ _ <<<"$r"
            pidfile=".tpu-logs/${name}.create.pid"
            if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
                echo "[detach] $name: already polling (pid $(cat "$pidfile"))"
                continue
            fi
            nohup bash -c "
                $(declare -f create_one)
                $(declare -f launch_one)
                $(declare -f create_and_launch_one)
                PROJECT='$PROJECT'
                create_and_launch_one '$r'
            " >/dev/null 2>&1 &
            echo $! > "$pidfile"
            echo "[detach] $name: pid $(cat "$pidfile") (logs .tpu-logs/${name}.{create,launch}.log)"
        done
        echo
        echo "All create+launch loops detached. Check with:"
        echo "  $0 --status"
        echo "  tail -f .tpu-logs/*.launch.log"
        ;;
    --status)
        status_all
        echo
        echo "Active retry-loop pids:"
        for f in .tpu-logs/*.create.pid; do
            [[ -f "$f" ]] || continue
            pid=$(cat "$f")
            name=$(basename "$f" .create.pid)
            if kill -0 "$pid" 2>/dev/null; then
                echo "  $name: pid $pid (running)"
            else
                echo "  $name: pid $pid (exited)"
            fi
        done
        ;;
    --launch)
        for r in "${ROWS[@]}"; do launch_one "$r" & done; wait
        ;;
    --delete)
        for r in "${ROWS[@]}"; do delete_one "$r" & done; wait
        # Also kill any retry loops still running.
        shopt -s nullglob
        for f in .tpu-logs/*.create.pid; do
            kill "$(cat "$f")" 2>/dev/null || true
            rm -f "$f"
        done
        shopt -u nullglob
        ;;
    sdm-*-wristband)
        # Single-VM mode: act on just this one. Defaults to --launch.
        sub="${2:---launch}"
        row=$(row_for "$1") || { echo "Unknown VM $1" >&2; exit 2; }
        case "$sub" in
            --create) create_one "$row" ;;
            --launch) launch_one "$row" ;;
            --delete) delete_one "$row" ;;
            *) echo "Unknown sub $sub" >&2; exit 2 ;;
        esac
        ;;
    *)
        echo "Usage: $0 {--create|--status|--launch|--delete|<vm-name> [--create|--launch|--delete]}" >&2
        exit 2
        ;;
esac
