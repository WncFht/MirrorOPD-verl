#!/usr/bin/env bash
# 指令遵循 OPD：UltraData-IF teacher，MultiIF 第一轮训练，三轮评测单独运行。
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export DOMAIN=if
exec bash "$SCRIPT_DIR/launchers/_run_single_teacher_fsdp2.sh" "$@"
