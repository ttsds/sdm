#!/usr/bin/env bash
# Shared bootstrap for every per-experiment TPU run script.
# Idempotent: clones the repo if missing, syncs deps via uv, sources .env,
# wires up wandb + xla env vars. The caller sets EXPERIMENT and CONFIG.

set -uo pipefail

: "${EXPERIMENT:?run script must export EXPERIMENT}"
: "${CONFIG:?run script must export CONFIG (path under configs/)}"

REPO_URL="${SDM_REPO_URL:-https://github.com/ttsds/sdm.git}"
REPO_DIR="${HOME}/sdm"

# 1. Clone or pull
if [[ ! -d "$REPO_DIR/.git" ]]; then
    git clone "$REPO_URL" "$REPO_DIR"
else
    git -C "$REPO_DIR" fetch --quiet origin && git -C "$REPO_DIR" pull --ff-only
fi
cd "$REPO_DIR"

# 2. Secrets — .env must be uploaded once via:
#       gcloud compute tpus tpu-vm scp .env <tpu>:~/sdm/.env --zone=<zone>
if [[ ! -f .env ]]; then
    echo "[$EXPERIMENT] .env missing at $REPO_DIR/.env -- upload it first" >&2
    exit 2
fi
set -a; source .env; set +a

# 3. Install uv if needed (some TPU images don't have it)
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 4. Sync deps. `tpu` extra is a placeholder; torch_xla wheels still install
# separately because they pin torch versions.
uv sync --extra dev --extra tpu --extra tracking

# 5. torch_xla — install if missing. Pinning to whatever torch ships in the lockfile.
if ! uv run python -c "import torch_xla" 2>/dev/null; then
    TORCH_VER=$(uv run python -c "import torch; print(torch.__version__.split('+')[0])")
    uv pip install "torch~=${TORCH_VER}" torch_xla \
        -f https://storage.googleapis.com/libtpu-releases/index.html
fi

# 6. XLA + wandb env
export PJRT_DEVICE=TPU
export PT_XLA_DEBUG_LEVEL=0
export WANDB_PROJECT="${WANDB_PROJECT:-sdm}"
export WANDB_RUN_NAME="${EXPERIMENT}-10pct"
# WANDB_API_KEY comes from .env

echo "[$EXPERIMENT] env ready; launching run for $CONFIG"
