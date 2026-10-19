#!/usr/bin/env bash
# 代码 OPD：DeepCoder teacher，TACO 训练，LiveCodeBench v6 验证。
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export DOMAIN=code
exec bash "$SCRIPT_DIR/_run_single_teacher_fsdp2.sh" "$@"
