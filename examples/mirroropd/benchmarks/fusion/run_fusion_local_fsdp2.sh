#!/usr/bin/env bash
# Explicit single-GPU development allocation; never attach to another Ray job.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../.." && pwd)
WORKSPACE=$(dirname -- "$REPO")
EXPECTED_UUID=GPU-76f3510f-d1e3-46ec-7cd6-a339d3a9d933
ACTUAL_UUID=$(nvidia-smi --id=7 --query-gpu=uuid --format=csv,noheader)
[[ $ACTUAL_UUID == "$EXPECTED_UUID" ]] || { echo "GPU 7 UUID changed; refusing allocation" >&2; exit 2; }
mkdir -p "$WORKSPACE/runs/vllm-fusion-20261029"
exec 9>"$WORKSPACE/runs/vllm-fusion-20261029/gpu7.lock"
flock -n 9 || { echo "A fusion GPU 7 run already owns this allocation" >&2; exit 2; }
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | rg -Fxq "$EXPECTED_UUID"; then
    echo "GPU 7 has an existing compute process; refusing to interfere" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES=7 STUDENT_GPUS=1 RAY_ADDRESS=local
export FUSION_RESOURCE_PARENT=$$
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_PLUGINS=mirroropd_fusion
export VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1
unset PYTHONDONTWRITEBYTECODE
export PYTHONPYCACHEPREFIX=/root/.cache/mirroropd-fusion/pycache
export PYTHON_BIN="$WORKSPACE/envs/mirroropd-fusion/bin/python"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export VERL_USE_UV=0
export SMOKE_RAY_TMPDIR="/tmp/mirroropd-vllm-gpu7-$$"
export LD_LIBRARY_PATH="$WORKSPACE/cache/fusion-runtime/cuda-compat-12-9/usr/local/cuda-12.9/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$REPO"
if (( $# == 0 )); then
    echo "Usage: bash run_fusion_local_fsdp2.sh <command> [args...]" >&2
    exit 2
fi
status=0
"$@" || status=$?
# Ray/vLLM children may release CUDA contexts shortly after their driver exits.
# Wait only: never kill another process or treat a failed NVML query as idle.
for attempt in {1..120}; do
    processes=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader) || exit 2
    if ! rg -Fxq "$EXPECTED_UUID" <<< "$processes"; then
        exit "$status"
    fi
    sleep 0.5
done
echo "GPU 7 is still occupied after driver exit; refusing the next stage" >&2
if (( status == 0 )); then status=2; fi
exit "$status"
