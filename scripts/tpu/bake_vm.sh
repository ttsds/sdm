#!/usr/bin/env bash
# Bake the SDM environment onto a freshly provisioned TPU VM.
# Run this on the VM (worker=all). Idempotent.
#
# Detects the TPU generation from /sys and installs the matching
# torch / torch_xla wheels.
set -euo pipefail

# Detect TPU generation.
ACCEL_TYPE="$(curl -sf -H 'Metadata-Flavor: Google' \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type \
    || echo unknown)"
echo "[bake] accelerator-type=$ACCEL_TYPE"

case "$ACCEL_TYPE" in
    v4-*)        TPU_GEN="tpuv4" ;;
    v5litepod-*) TPU_GEN="tpuv5e" ;;
    v6e-*)       TPU_GEN="tpuv6e" ;;
    *)           echo "[bake] unknown generation, defaulting to tpuv4"; TPU_GEN="tpuv4" ;;
esac
echo "[bake] generation=$TPU_GEN"

# uv (fast Python installer) — already on most TPU VMs; install if missing.
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Use Python 3.11 (most stable torch_xla support); 3.14 is too new for prebuilt wheels.
PYBIN="${PYBIN:-python3.11}"
if ! command -v "$PYBIN" >/dev/null; then
    sudo apt-get update -y && sudo apt-get install -y python3.11 python3.11-venv
fi

cd "$(dirname "$0")/../.."
if [[ ! -d .venv ]]; then
    uv venv --python "$PYBIN" .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Install torch + torch_xla matching the TPU generation.
# - torch_xla 2.5 pinned to a libtpu-nightly version that is no longer on PyPI
#   (resolution failure). 2.6+ moved to stable libtpu pulled via torch_xla[tpu]
#   from the libtpu-releases index. v6e support requires >=2.6; we use 2.7
#   which pins libtpu 0.0.11.1 (stable).
uv pip install --upgrade pip
uv pip install 'torch~=2.7.0' 'torch_xla[tpu]~=2.7.0' \
    -f https://storage.googleapis.com/libtpu-releases/index.html \
    -f https://storage.googleapis.com/libtpu-wheels/index.html

# SDM + TPU-safe extras (skip teachers — those run on CPU/GPU host, not TPU).
uv pip install -e '.[tracking]'

# Sanity: import torch_xla and list devices.
python - <<'PY'
import os
os.environ.setdefault("PJRT_DEVICE", "TPU")
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
print("[bake] torch_xla:", torch_xla.__version__)
print("[bake] xla device:", xm.xla_device())
print("[bake] world size:", xr.world_size())
PY

echo "[bake] done."
