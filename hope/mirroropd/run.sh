#!/usr/bin/env bash
# Hope 单机 8 卡入口；实际训练参数由 examples/mirroropd/ 配方维护。
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

# 保留调度器设置的设备编号或 UUID；未设置时使用任务内的 8 个逻辑设备。
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export STUDENT_GPUS=4
export TEACHER_GPUS=4
export ROLLOUT_TP=1
export TEACHER_TP=1
export PYTHON_BIN="$REPO_ROOT/.venv-native/bin/python"
export VERL_USE_UV=0
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# 预览命令和配置时不访问 GPU，也不解压依赖或创建训练目录。
if [[ ${DRY_RUN:-0} != 1 && ${CONFIG_ONLY:-0} != 1 ]]; then
    make native-check PYTHON="$PYTHON_BIN" NATIVE_PYTHON="$PYTHON_BIN"
    "$PYTHON_BIN" - <<'PY'
import torch

count = torch.cuda.device_count()
if count != 8:
    raise RuntimeError(f"Hope job requires exactly 8 visible GPUs, got {count}")
for index in range(count):
    device = torch.device("cuda", index)
    value = torch.ones((2, 2), device=device)
    result = value @ value
    torch.cuda.synchronize(device)
    if result[0, 0].item() != 2:
        raise RuntimeError(f"GPU {index}: unexpected matrix product")
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}; CUDA matrix product passed", flush=True)
PY
fi

exec bash "$REPO_ROOT/examples/mirroropd/run_deepseek_r1_1_5b_fsdp2.sh" "$@"
