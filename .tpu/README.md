# `.tpu/` — TPU provisioning helpers

Per-experiment scripts that bring up a v6e-8 spot TPU in `europe-west4-a`,
SCP a `.env` onto it, and launch the configured run inside `tmux -s sdm`.

Secrets (`WANDB_API_KEY`, `HF_TOKEN`, optional `SDM_GCP_PROJECT`,
`SDM_TPU_ZONE`, `SDM_TPU_ACCEL`, `SDM_TPU_RUNTIME`) live in the repo-root
`.env` (gitignored). `_lib.sh` sources it; the per-script files are
commit-safe and contain no secrets.

## Files

| File | Purpose |
|---|---|
| `_lib.sh` | shared helpers: sources `.env`, defines `poll_create`, `wait_ready`, `push_env`, `launch_run`, `provision`. |
| `provision_<short>.sh` | one per experiment; sets `NAME` + `CONFIG`, sources `_lib.sh`, calls `provision`. |
| `provision_probes.sh` | brings up `sdm-probes` and runs `scripts/run_probes.sh` instead of `run_finetune.sh` (overrides `LAUNCH_CMD`). |
| `provision_all.sh` | nohup fan-out across all 8 finetunes; writes `logs/<short>.log` per job. |
| `teardown.sh` | `gcloud ... tpu-vm delete` for every known VM and kills local poll loops. |
| `logs/` | runtime logs (gitignored). |

## Mapping

| Script | TPU name | Config / driver |
|---|---|---|
| `provision_dvector.sh`        | `sdm-dvector`        | `configs/finetune_dvector.yaml` |
| `provision_wespeaker.sh`      | `sdm-wespeaker`      | `configs/finetune_wespeaker.yaml` |
| `provision_pitch.sh`          | `sdm-pitch`          | `configs/finetune_pitch.yaml` |
| `provision_mpm.sh`            | `sdm-mpm`            | `configs/finetune_mpm.yaml` |
| `provision_speaking_rate.sh`  | `sdm-speaking-rate`  | `configs/finetune_speaking_rate.yaml` |
| `provision_w2v2_asr.sh`       | `sdm-w2v2-asr`       | `configs/finetune_w2v2_asr.yaml` |
| `provision_mwhisper.sh`       | `sdm-mwhisper`       | `configs/finetune_mwhisper.yaml` |
| `provision_emotion2vec.sh`    | `sdm-emotion2vec`    | `configs/finetune_emotion2vec.yaml` |
| `provision_probes.sh`         | `sdm-probes`         | `scripts/run_probes.sh` (cross-teacher probe matrix) |

The mHuBERT-on-XLS-R run (`sdm-xlsr`, `configs/finetune_xlsr.yaml`) was the
first finetune and is provisioned ad-hoc; the others followed the template
above.

## Usage

```bash
# Bring up all 8 finetunes in parallel (background, polls until capacity).
./.tpu/provision_all.sh

# Or one at a time:
./.tpu/provision_pitch.sh

# After all finetunes converge, run the probe phase:
./.tpu/provision_probes.sh

# Watch progress:
tail -f .tpu/logs/*.log
gcloud compute tpus tpu-vm list \
    --project="$SDM_GCP_PROJECT" \
    --zone="${SDM_TPU_ZONE:-europe-west4-a}" \
    --filter='name~^sdm-'

# Tear down:
./.tpu/teardown.sh
```

## How a provision script flows

1. `_lib.sh` sources `.env` (fails loudly if missing).
2. `poll_create` retries `gcloud compute tpus tpu-vm create $NAME --spot`
   until v6e-8 capacity opens; recognised retryable errors don't sleep
   excessively, unrecognised errors retry with a warning.
3. `wait_ready` polls `gcloud ... describe` for `state=READY` and recreates
   on `PREEMPTED`/`TERMINATED`/`FAILED`.
4. `push_env` SCPs an in-memory `.env.sdm` (just `WANDB_API_KEY`,
   `HF_TOKEN`, `SDM_GCP_PROJECT`) onto the VM at `~/.env.sdm`.
5. `launch_run` SSHes in, clones (or pulls) the repo, copies `.env.sdm` to
   `~/sdm/.env`, and starts the configured `LAUNCH_CMD` inside `tmux -s sdm`
   (default: `CONFIG=$CONFIG bash scripts/run_finetune.sh`; `provision_probes.sh`
   overrides this with `bash scripts/run_probes.sh`).
6. `scripts/run_finetune.sh` (or `run_probes.sh`) sources `_tpu_common.sh`
   on the VM to install the matching `torch_xla` wheel and resume from the
   latest GCS checkpoint.

## Adding a new experiment

1. Drop a `configs/finetune_<short>.yaml` into the repo (matching the field
   shape of the existing configs).
2. Copy any existing `provision_<short>.sh`, change `NAME` and `CONFIG`,
   commit it.
3. Add the script to `SCRIPTS=(...)` in `provision_all.sh` if you want it in
   the fanout, and to `NAMES=(...)` in `teardown.sh` so cleanup catches it.
