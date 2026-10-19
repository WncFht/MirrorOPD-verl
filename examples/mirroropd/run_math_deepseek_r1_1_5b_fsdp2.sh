#!/usr/bin/env bash
# 数学 OPD：DAPO 训练，MATH-500 / AMC23 / AIME25 验证。
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export DOMAIN=math
exec bash "$SCRIPT_DIR/_run_single_teacher_fsdp2.sh" "$@"
