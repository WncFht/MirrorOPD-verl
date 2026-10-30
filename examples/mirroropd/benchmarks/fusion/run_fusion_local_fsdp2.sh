#!/usr/bin/env bash
# Explicit local allocation; the guardian owns locks outside model/monitor children.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../.." && pwd)
WORKSPACE=$(dirname -- "$REPO")
exec "$WORKSPACE/envs/mirroropd-fusion/bin/python" "$HERE/resource_guard.py" \
    run --repo "$REPO" --gpus "${FUSION_GPUS:-7}" -- "$@"
