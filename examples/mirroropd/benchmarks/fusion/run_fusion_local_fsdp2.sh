#!/usr/bin/env bash
# Explicit local development allocation; never attach to another Ray job.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../.." && pwd)
WORKSPACE=$(dirname -- "$REPO")
FUSION_GPUS=${FUSION_GPUS:-7}
case "$FUSION_GPUS" in 6|7|6,7) ;; *) echo "Only the allocated GPU sets 6, 7, or 6,7 are allowed" >&2; exit 2 ;; esac
IFS=',' read -r -a DEVICES <<< "$FUSION_GPUS"
UUIDS=()
mkdir -p "$WORKSPACE/runs/vllm-fusion-20261029"
for device in "${DEVICES[@]}"; do
    case "$device" in
        6)
            expected=GPU-44bf7e96-2d92-fcfd-7da0-471d76517118
            exec 8>"$WORKSPACE/runs/vllm-fusion-20261029/gpu6.lock"
            flock -n 8 || { echo "GPU 6 allocation is already locked" >&2; exit 2; }
            ;;
        7)
            expected=GPU-76f3510f-d1e3-46ec-7cd6-a339d3a9d933
            exec 9>"$WORKSPACE/runs/vllm-fusion-20261029/gpu7.lock"
            flock -n 9 || { echo "GPU 7 allocation is already locked" >&2; exit 2; }
            ;;
    esac
    actual=$(nvidia-smi --id="$device" --query-gpu=uuid --format=csv,noheader)
    [[ $actual == "$expected" ]] || { echo "GPU $device UUID changed; refusing allocation" >&2; exit 2; }
    UUIDS+=("$expected")
done
allocated_busy() {
    local processes uuid
    processes=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader) || return 2
    for uuid in "${UUIDS[@]}"; do
        if rg -Fxq "$uuid" <<< "$processes"; then return 0; fi
    done
    return 1
}
if allocated_busy; then
    echo "An allocated GPU has an existing compute process; refusing to interfere" >&2
    exit 2
else
    query_status=$?
    (( query_status == 1 )) || exit 2
fi
export CUDA_VISIBLE_DEVICES="$FUSION_GPUS" STUDENT_GPUS=${#DEVICES[@]} RAY_ADDRESS=local
export FUSION_RESOURCE_PARENT=$$
export VERL_ROLLOUT_IPC_NAMESPACE="fusion-${FUSION_GPUS//,/}-${BASHPID}-$(date +%s)"
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_PLUGINS=mirroropd_fusion
export VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1
unset PYTHONDONTWRITEBYTECODE
export PYTHONPYCACHEPREFIX=/root/.cache/mirroropd-fusion/pycache
export PYTHON_BIN="$WORKSPACE/envs/mirroropd-fusion/bin/python"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export VERL_USE_UV=0
export SMOKE_RAY_TMPDIR="/tmp/mirroropd-fusion-${FUSION_GPUS//,/}-$$"
export LD_LIBRARY_PATH="$WORKSPACE/cache/fusion-runtime/cuda-compat-12-9/usr/local/cuda-12.9/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$REPO"
if (( $# == 0 )); then
    echo "Usage: FUSION_GPUS=7 bash run_fusion_local_fsdp2.sh <command> [args...]" >&2
    exit 2
fi
status=0
"$@" || status=$?
# Wait for CUDA cleanup only; never kill another process or treat NVML failure as idle.
for attempt in {1..120}; do
    if allocated_busy; then
        sleep 0.5
    else
        query_status=$?
        (( query_status == 1 )) || exit 2
        exit "$status"
    fi
done
echo "Allocated GPUs remain occupied after driver exit; refusing the next stage" >&2
if (( status == 0 )); then status=2; fi
exit "$status"
