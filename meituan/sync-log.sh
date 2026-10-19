#!/usr/bin/env bash
set -eo pipefail

# Pack W&B runs on the developer machine; unpack the downloaded files locally.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${MIRROROPD_PYTHON:-python3}" "$SCRIPT_DIR/transfer.py" logs "$@"
