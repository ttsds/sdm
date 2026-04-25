#!/usr/bin/env bash
# Smoke test the SDM pipeline on a TPU VM. Run after scripts/tpu/bake_vm.sh.
# Uses the synthetic dataset path so no network/dataset access required.
set -euo pipefail

cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source .venv/bin/activate

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

export PJRT_DEVICE=TPU
# NOTE: do NOT set XLA_USE_BF16=1 — it downcasts *all* fp32 ops (incl. softmax /
# layernorm) to bf16 and produces NaNs at init for large vocabs. Use targeted
# autocast inside the model instead if bf16 compute is desired.
export WANDB_DISABLED="${WANDB_DISABLED:-true}"   # opt-in; default off for smoke

# Tiny synthetic config: 50 steps, model SDM_SMALL.
cat > /tmp/sdm_smoke.yaml <<'YAML'
model:
  fsq_vocab_size: 65536
  hidden_size: 384
  num_hidden_layers: 4
  num_attention_heads: 6
  intermediate_size: 1536
  max_position_embeddings: 256
  position_buckets: 64

data:
  repo_id: synthetic
  split: train
  max_length: 256
  min_codes: 16
  streaming: true
  shuffle_buffer: 0
  seed: 0

mlm:
  mask_prob: 0.15
  mask_replace_prob: 0.8
  random_replace_prob: 0.1

train:
  batch_size: 2
  grad_accum: 1
  lr: 1.0e-4
  weight_decay: 0.01
  warmup_steps: 5
  total_steps: 50
  log_every: 5
  ckpt_every: 25
  ckpt_dir: /tmp/sdm_smoke_ckpts
  fsdp: false
  resume_from_latest: false
  seed: 0
YAML

echo "[smoke] running synthetic pretrain (50 steps)..."
python -m sdm.train.run_pretrain --config /tmp/sdm_smoke.yaml --synthetic

echo "[smoke] success."
ls -la /tmp/sdm_smoke_ckpts || true
