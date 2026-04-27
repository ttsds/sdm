# TPU resources (TRC grant)

GCP project: `ml-edinburgh`. All slices are managed via
`gcloud compute tpus queued-resources` (use `--best-effort` for spot,
`--reserved` for on-demand).

## Inventory

| Pool | Chips | Generation | Zone | Class | Role |
|---|---|---|---|---|---|
| `sdm-pretrain-primary` | 32 | v4 | us-central2-b | on-demand | Primary pretrain (250 k steps, baseline) |
| `sdm-pretrain-standby` | 32 | v4 | us-central2-b | spot | Warm standby for primary; teacher-cache overflow |
| `sdm-ablation-pretrain` | 64 | v6e | us-east1-d | spot | Ablation pretrain (wristband-Gaussian variant) |
| `sdm-finetune-headline` | 64 | v6e | europe-west4-a | spot | Headline finetunes (`generic`, `intelligibility`) + ablation twins |
| `sdm-teacher-cache` | 64 | v5e | europe-west4-b | spot | Teacher cache extraction (4 factors in parallel) |
| `sdm-finetune-extra` | 64 | v5e | us-central1-a | spot | Speaker / prosody finetunes (post-port); eval sweeps |

Image baking and resume strategy live in
[scripts/run_finetune.sh](scripts/run_finetune.sh),
[scripts/_tpu_common.sh](scripts/_tpu_common.sh), and
[sdm/train/xla_utils.py](sdm/train/xla_utils.py).

## GCS layout

| Bucket | Region | Use |
|---|---|---|
| `gs://sdm-ckpts` | multi-region (US+EU dual) | All checkpoints (read by every slice) |
| `gs://sdm-cache-usc2` | us-central2 | Teacher cache shards (used by us-central2-b slices) |
| `gs://sdm-cache-use1` | us-east1 | Teacher cache shards (us-east1-d) |
| `gs://sdm-cache-euw4` | europe-west4 | Teacher cache shards (europe-west4-{a,b}) |
| `gs://sdm-cache-usc1` | us-central1 | Teacher cache shards (us-central1-a) |

Same-region reads avoid egress charges.

## Provisioning commands (sketch)

```bash
# On-demand v4-64 (primary pretrain).
gcloud compute tpus queued-resources create sdm-pretrain-primary \
    --node-id=sdm-pretrain-primary \
    --project=ml-edinburgh \
    --zone=us-central2-b \
    --accelerator-type=v4-64 \
    --runtime-version=tpu-ubuntu2204-base \
    --reserved

# Spot v6e-64 (ablation pretrain).
gcloud compute tpus queued-resources create sdm-ablation-pretrain \
    --node-id=sdm-ablation-pretrain \
    --project=ml-edinburgh \
    --zone=us-east1-d \
    --accelerator-type=v6e-64 \
    --runtime-version=v2-alpha-tpuv6e \
    --best-effort
```

`--runtime-version` differs per generation; bake one image per gen with
`uv` + this repo + the matching `torch` / `torch_xla` wheel pre-installed
to avoid spot-churn reinstalls.
