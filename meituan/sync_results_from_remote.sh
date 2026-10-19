#!/usr/bin/env bash
set -eo pipefail

# Result archives are transferred manually between the three checkouts.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${MIRROROPD_PYTHON:-python3}" "$SCRIPT_DIR/transfer.py" results "$@"
