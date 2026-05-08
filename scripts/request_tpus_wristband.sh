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
V5E_ZONE="${SDM_TPU_ZONE_V5E:-us-central1-a}"
V6E_RUNTIME="v2-alpha-tpuv6e"
V5E_RUNTIME="v2-alpha-tpuv5-lite"

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
    "sdm-xlsr-wristband|configs/finetune_xlsr_wristband.yaml|v5e-8|$V5E_ZONE|$V5E_RUNTIME"
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
    echo "[create] $name ($accel in $zone)"
    # --best-effort = spot for queued-resources. Returns immediately while
    # the request waits for capacity server-side.
    gcloud compute tpus queued-resources create "$name" \
        --node-id="$name" \
        --project="$PROJECT" \
        --zone="$zone" \
        --accelerator-type="$accel" \
        --runtime-version="$runtime" \
        --best-effort \
        --async || echo "[create] $name: request failed (already exists?)"
}

delete_one() {
    local row="$1"
    IFS='|' read -r name _ _ zone _ <<<"$row"
    echo "[delete] $name (zone=$zone)"
    # Force-delete the underlying VM first if it's running, then drop the
    # queued resource record.
    gcloud compute tpus tpu-vm delete "$name" \
        --project="$PROJECT" --zone="$zone" --quiet 2>/dev/null || true
    gcloud compute tpus queued-resources delete "$name" \
        --project="$PROJECT" --zone="$zone" --force --quiet 2>/dev/null || true
}

status_all() {
    declare -A ZONES
    for r in "${ROWS[@]}"; do
        IFS='|' read -r name _ _ zone _ <<<"$r"
        ZONES["$zone"]=1
    done
    for z in "${!ZONES[@]}"; do
        echo "=== $z ==="
        gcloud compute tpus queued-resources list \
            --project="$PROJECT" --zone="$z" \
            --filter="name~-wristband$" \
            --format="table(name,state.state,acceleratorType)" 2>/dev/null
    done
}

launch_one() {
    local row="$1"
    IFS='|' read -r name cfg _ zone _ <<<"$row"
    local ssh="gcloud compute tpus tpu-vm ssh $name --project=$PROJECT --zone=$zone --worker=all"
    local scp="gcloud compute tpus tpu-vm scp --project=$PROJECT --zone=$zone --worker=all"

    echo "[launch] $name -> $cfg"
    # Wait for VM to exist (queued-resource ACTIVE).
    for i in $(seq 1 60); do
        if gcloud compute tpus tpu-vm describe "$name" \
                --project="$PROJECT" --zone="$zone" \
                --format="value(state)" 2>/dev/null | grep -q READY; then
            break
        fi
        echo "[launch] $name not READY yet (attempt $i); sleeping 30s"
        sleep 30
    done

    # 1. Make sure ~/sdm exists (clone if needed) and has the latest tip.
    $ssh --command "test -d ~/sdm/.git || git clone https://github.com/ttsds/sdm.git ~/sdm; cd ~/sdm && git fetch --quiet origin && git reset --hard origin/main"

    # 2. Upload .env (contains WANDB_API_KEY, HF_TOKEN, SDM_GCP_PROJECT).
    $scp .env "$name:~/sdm/.env"

    # 3. Start training in a detached tmux session so SSH disconnect doesn't
    #    kill it. Logs land in ~/sdm/train.log; tail with:
    #      $ssh --command "tail -f ~/sdm/train.log"
    $ssh --command "cd ~/sdm && tmux kill-session -t sdm 2>/dev/null; tmux new -d -s sdm \"CONFIG=$cfg ./scripts/run_finetune.sh > train.log 2>&1\""
    echo "[launch] $name: started"
}

case "${1:-}" in
    --create)
        for r in "${ROWS[@]}"; do create_one "$r" & done; wait
        echo
        echo "All 9 requests queued. Watch with: $0 --status"
        ;;
    --status)
        status_all
        ;;
    --launch)
        for r in "${ROWS[@]}"; do launch_one "$r" & done; wait
        ;;
    --delete)
        for r in "${ROWS[@]}"; do delete_one "$r" & done; wait
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
