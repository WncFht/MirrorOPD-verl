#!/usr/bin/env bash
set -eo pipefail

# Code snapshots travel through manual upload/download, without a remote shell.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${MIRROROPD_PYTHON:-python3}" "$SCRIPT_DIR/code_archive.py" "$@"
