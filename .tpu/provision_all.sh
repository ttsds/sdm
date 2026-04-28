#!/usr/bin/env bash
# Fan out the per-finetune provision scripts in parallel. Each writes to
# .tpu/logs/<name>.log; failures are non-fatal so the rest keep going.
# Ctrl-C to cancel — children inherit signal handling.
#
# Excludes provision_probes.sh: probes are run once after all finetunes
# converge, not in the fanout.

set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs

SCRIPTS=(
    provision_dvector.sh
    provision_wespeaker.sh
    provision_pitch.sh
    provision_mpm.sh
    provision_speaking_rate.sh
    provision_w2v2_asr.sh
    provision_mwhisper.sh
    provision_emotion2vec.sh
)

pids=()
for s in "${SCRIPTS[@]}"; do
    name="${s#provision_}"; name="${name%.sh}"
    log="logs/${name}.log"
    echo "[fanout] launching $s -> $log"
    nohup bash "$s" >>"$log" 2>&1 &
    pids+=("$!:$name")
done

echo
echo "[fanout] launched ${#pids[@]} provision jobs:"
for entry in "${pids[@]}"; do
    pid="${entry%%:*}"
    name="${entry##*:}"
    echo "  pid=$pid  name=$name  log=logs/${name}.log"
done
echo
echo "[fanout] tail all logs with:    tail -f logs/*.log"
echo "[fanout] check VM status with:  gcloud compute tpus tpu-vm list --project=\"\$SDM_GCP_PROJECT\" --zone=\"\${SDM_TPU_ZONE:-europe-west4-a}\" --filter='name~^sdm-'"
echo "[fanout] kill all with:         pkill -f provision_  (or ./teardown.sh)"
