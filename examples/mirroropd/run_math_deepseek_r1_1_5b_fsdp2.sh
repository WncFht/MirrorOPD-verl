#!/usr/bin/env bash
# 数学 OPD：DAPO 训练，MATH-500 / AMC23 / AIME25 验证。
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export DOMAIN=math
# Validated on four H800s with 128 prompts x 16 responses; keep the
# evaluation queue supplied without changing the training batch or budget.
# Other domains retain the common launcher's defaults. Explicit overrides win.
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
exec bash "$SCRIPT_DIR/_run_single_teacher_fsdp2.sh" "$@"
