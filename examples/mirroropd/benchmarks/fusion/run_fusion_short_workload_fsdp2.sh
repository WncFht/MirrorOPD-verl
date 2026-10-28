#!/usr/bin/env bash
# Explicit non-thinking protocol; actual natural lengths determine workload coverage.
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../.." && pwd)
WORKSPACE=$(dirname -- "$REPO")
PAIR=${1:?Choose dense or moe}
PREFIX=${2:-fusion-nonthinking-$(date +%Y%m%d-%H%M%S)}
case "$PAIR" in
    dense) TEACHER=$WORKSPACE/model/Qwen3-8B; MICRO=16 ;;
    moe) TEACHER=$WORKSPACE/model/Qwen3-30B-A3B-Instruct-2507; MICRO=8 ;;
    *) exit 2 ;;
esac
OUT=$WORKSPACE/runs/rollout-fusion-20261028/$PREFIX
mkdir -p "$OUT"
cd "$REPO"
export CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 PYTHONPATH=$REPO
export LD_LIBRARY_PATH=$WORKSPACE/cache/fusion-runtime/cuda-compat-12-9/usr/local/cuda-12.9/compat:${LD_LIBRARY_PATH:-}
PYTHON_BIN=${PYTHON_BIN:-/tmp/mirroropd-fusion-venv/bin/python}
run() {
    local name=$1
    shift
    if [[ -e $OUT/$name ]]; then
        echo "Refusing to overwrite $OUT/$name" >&2
        exit 2
    fi
    "$PYTHON_BIN" "$HERE/benchmark.py" \
        --student "$WORKSPACE/model/Qwen3-1.7B" --teacher "$TEACHER" \
        --prompt-model "$WORKSPACE/model/Qwen3-1.7B-Base" \
        --chat-template-kwargs '{"enable_thinking":false}' \
        --data "$WORKSPACE/data/math/DAPO/train.parquet" --out "$OUT/$name" \
        --batch 64 --cap 7168 --alpha 0.5 --microbatch "$MICRO" --compact 2 --repeats 3 \
        "$@" > "$OUT/$name.log" 2>&1
}
{
run "$PAIR-nonthinking-smoke" --batch 8 --cap 512 --microbatch 2 --repeats 1 \
    --stop-diagnostics --cuda-graph --flash-kv-splits 1
run "$PAIR-nonthinking-reference"
run "$PAIR-nonthinking-optimized" --compact 0 2 --cuda-graph --flash-kv-splits 1
mapfile -t POINTS < <("$PYTHON_BIN" - "$OUT/$PAIR-nonthinking-optimized" <<'PY'
import json
import sys
from pathlib import Path

lengths = []
for path in Path(sys.argv[1]).glob("rank-*.jsonl"):
    for line in path.open():
        row = json.loads(line)
        if row["kind"] == "measurement" and row["repeat"] == 0 and row["compact"] == 2:
            lengths.extend(row["lengths"])
assert len(lengths) == 64
last = max(lengths) - 1
for point in sorted({p for p in [0, 1, 2, 128, 512, 1024, 2048, 4096, 6144, 7167, last] if 0 <= p <= last}):
    print(point)
PY
)
(( ${#POINTS[@]} > 0 ))
"$PYTHON_BIN" "$HERE/probe_history.py" --trajectory "$OUT/$PAIR-nonthinking-optimized" \
    --out "$OUT/$PAIR-nonthinking-history" --points "${POINTS[@]}" \
    > "$OUT/$PAIR-nonthinking-history.log" 2>&1
exit
}
