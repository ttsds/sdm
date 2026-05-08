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

# 3b. Force a Python 3.11 venv. The TPU base image ships system Python 3.10,
# but torch_xla 2.7 wheels for libtpu are built against 3.10/3.11/3.12 only,
# and we hit a pernicious ABI mismatch (`undefined symbol:
# _ZNK3c1010TensorImpl39sym_is_non_overlapping_and_dense_customEv`) when uv
# resolves a torch_xla wheel that was actually built against a newer torch
# than our `~=2.7.0` pin. Sticking to 3.11 + explicit pinning below avoids it.
PYBIN="${SDM_PYTHON:-python3.11}"
if ! command -v "$PYBIN" >/dev/null 2>&1; then
    sudo apt-get update -y && sudo apt-get install -y python3.11 python3.11-venv
fi
if [[ ! -d .venv ]] || ! .venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11)' 2>/dev/null; then
    rm -rf .venv
    uv venv --python "$PYBIN" .venv
fi

# 4. Sync deps. `tpu` extra is a placeholder; torch / torch_xla install in
# the next step with explicit `~=2.7.0` pins to keep their ABIs aligned.
uv sync --extra dev --extra tpu --extra tracking

# 5. torch + torch_xla — pin both to the same minor (2.7) and pull libtpu via
# the `[tpu]` extra. Re-run unconditionally; uv pip is idempotent and this
# overrides any drift introduced by `uv sync`.
uv pip install \
    'torch~=2.7.0' \
    'torchaudio~=2.7.0' \
    'torch_xla[tpu]~=2.7.0' \
    -f https://storage.googleapis.com/libtpu-releases/index.html \
    -f https://storage.googleapis.com/libtpu-wheels/index.html

# 5b. Smoke test: import torch_xla so we fail loudly here, not 30 s into
# the train loop.
uv run python -c "import torch, torch_xla; print('[bootstrap] torch', torch.__version__, 'torch_xla', torch_xla.__version__)"

# 5c. Teacher runtime deps. These run on the TPU host CPU (in dataloader
# workers or the no-grad teacher pass), not on the TPU itself, so they're
# safe to install alongside torch_xla. We install the union for all 7
# remaining teachers — cheap, and lets the same bootstrap serve every
# experiment without per-config branching.
#   - phonemizer + espeak-ng: g2p_speaking_rate
#   - pyworld:                pyworld_f0
#   - wespeaker-unofficial:   wespeaker_resnet34 (replaces pyannote.audio,
#                             which had recurring torch / huggingface_hub
#                             version conflicts with our torch 2.7 pin)
#   - onnxruntime:            transitive dep of silero-vad, which
#                             wespeaker-unofficial imports at package
#                             __init__ time even though we never call VAD
#   - funasr:                 emotion2vec (AutoModel wrapper -- canonical
#                             loader maintained by the model authors)
#   - masked_prosody_model:   mpm (git, no PyPI release)
#   - allosaurus:             allosaurus_speaking_rate (per-chunk syllable
#                             counts via IPA phone timestamps)
# transformers / huggingface-hub / soundfile / librosa are already in core.
if ! dpkg -s espeak-ng >/dev/null 2>&1; then
    sudo apt-get update -y && sudo apt-get install -y espeak-ng libespeak-ng1
fi
uv pip install \
    phonemizer \
    pyworld \
    wespeaker-unofficial \
    onnxruntime \
    funasr \
    allosaurus \
    'masked_prosody_model @ git+https://github.com/MiniXC/masked_prosody_model'

# 6. XLA + wandb env
export PJRT_DEVICE=TPU
export PT_XLA_DEBUG_LEVEL=0
export WANDB_PROJECT="${WANDB_PROJECT:-sdm}"
export WANDB_RUN_NAME="${EXPERIMENT}-10pct"
# WANDB_API_KEY comes from .env

echo "[$EXPERIMENT] env ready; launching run for $CONFIG"
