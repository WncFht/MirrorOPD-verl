#!/usr/bin/env bash
# Rebuild the isolated CUDA 12.9 environment without touching shared environments.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../.." && pwd)
WORKSPACE=$(dirname -- "$REPO")
ENV_DIR="$WORKSPACE/envs/mirroropd-fusion"
VLLM_WHEEL="$WORKSPACE/cache/fusion-runtime/vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
FLASH_WHEEL="$WORKSPACE/cache/fusion-runtime/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
[[ -f $VLLM_WHEEL && -f $FLASH_WHEEL ]] || { echo "Required audited CUDA 12.9 wheels are missing" >&2; exit 2; }
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    [[ -x $ENV_DIR/bin/python ]] || uv venv --python 3.12 "$ENV_DIR"
    uv pip install --python "$ENV_DIR/bin/python" --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --overrides "$HERE/requirements-overrides.txt" \
        -r "$HERE/requirements-runtime.txt" "$VLLM_WHEEL" "$FLASH_WHEEL"
    uv pip install --python "$ENV_DIR/bin/python" --no-deps -e "$REPO"
    uv pip freeze --python "$ENV_DIR/bin/python"
else
    echo "This setup requires DEVICE=gpu and VERL_USE_UV=1" >&2
    exit 2
fi
