#!/usr/bin/env bash
# Run one explicitly named four-GPU performance block; never overwrite evidence.
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../.." && pwd)
WORKSPACE=$(dirname -- "$REPO")
BLOCK=${1:?Choose dense-base, dense-reference, dense-post, moe-scan, moe-base, moe-reference, moe-post, layouts, replay, profile, or singles}
PREFIX=${2:-g7-$(date +%Y%m%d-%H%M%S)}
OUT=$WORKSPACE/runs/rollout-fusion-20261028/$PREFIX
mkdir -p "$OUT"
cd "$REPO"
export CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 PYTHONPATH=$REPO
export LD_LIBRARY_PATH=$WORKSPACE/cache/fusion-runtime/cuda-compat-12-9/usr/local/cuda-12.9/compat:${LD_LIBRARY_PATH:-}
PYTHON_BIN=${PYTHON_BIN:-/tmp/mirroropd-fusion-venv/bin/python}
BASE=$WORKSPACE/model/Qwen3-1.7B-Base
POST=$WORKSPACE/model/Qwen3-1.7B
DENSE=$WORKSPACE/model/Qwen3-8B
MOE=$WORKSPACE/model/Qwen3-30B-A3B-Instruct-2507
GRAPH=(--cuda-graph --flash-kv-splits 1)

run() {
    local name=$1 student=$2 teacher=$3
    shift 3
    if [[ -e $OUT/$name ]]; then
        echo "Refusing to overwrite $OUT/$name" >&2
        exit 2
    fi
    "$PYTHON_BIN" "$HERE/benchmark.py" --student "$student" --teacher "$teacher" --prompt-model "$BASE" \
        --data "$WORKSPACE/data/math/DAPO/train.parquet" --out "$OUT/$name" \
        --precision stable --attention flash_attention_2 --attention-dtype fp16 \
        --batch 64 --cap 7168 --compact 2 --repeats 3 "$@" > "$OUT/$name.log" 2>&1
}

# Parse the entire execution block before running long jobs.
{
case "$BLOCK" in
    dense-base)
        run dense-base-optimized "$BASE" "$DENSE" --microbatch 16 "${GRAPH[@]}"
        ;;
    dense-reference)
        run dense-base-reference "$BASE" "$DENSE" --microbatch 16
        ;;
    dense-post)
        run dense-post-smoke "$POST" "$DENSE" --cap 128 --batch 8 --microbatch 2 --repeats 1 \
            --stop-diagnostics "${GRAPH[@]}"
        run dense-post-reference "$POST" "$DENSE" --microbatch 16
        run dense-post-optimized "$POST" "$DENSE" --microbatch 16 --compact 0 2 "${GRAPH[@]}"
        ;;
    moe-scan)
        run moe-scan "$BASE" "$MOE" --cap 256 --microbatch 2 4 8 --repeats 3 "${GRAPH[@]}"
        ;;
    moe-base)
        run moe-base-optimized "$BASE" "$MOE" --microbatch 8 "${GRAPH[@]}"
        ;;
    moe-reference)
        if [[ -f $OUT/dense-base-optimized/summary.json && -f $OUT/moe-base-optimized/summary.json ]]; then
            "$PYTHON_BIN" "$HERE/probe_history.py" --trajectory "$OUT/dense-base-optimized" --points 0 1 2 \
                --out "$OUT/dense-history-smoke" > "$OUT/dense-history-smoke.log" 2>&1
            "$PYTHON_BIN" "$HERE/probe_history.py" --trajectory "$OUT/dense-base-optimized" \
                --out "$OUT/dense-history-probe" > "$OUT/dense-history-probe.log" 2>&1
            "$PYTHON_BIN" "$HERE/probe_history.py" --trajectory "$OUT/moe-base-optimized" \
                --out "$OUT/moe-history-probe" > "$OUT/moe-history-probe.log" 2>&1
        fi
        run moe-base-reference "$BASE" "$MOE" --microbatch 8
        ;;
    moe-post)
        run moe-post-smoke "$POST" "$MOE" --cap 128 --batch 8 --microbatch 2 --repeats 1 \
            --stop-diagnostics "${GRAPH[@]}"
        run moe-post-reference "$POST" "$MOE" --microbatch 8
        run moe-post-optimized "$POST" "$MOE" --microbatch 8 --compact 0 2 "${GRAPH[@]}"
        ;;
    layouts)
        run dense-paired-smoke "$BASE" "$DENSE" --layout paired --microbatch 32 --cap 128 --repeats 1 "${GRAPH[@]}"
        run moe-paired-smoke "$BASE" "$MOE" --layout paired --microbatch 16 --cap 128 --repeats 1 "${GRAPH[@]}"
        run dense-base-paired "$BASE" "$DENSE" --layout paired --microbatch 32 "${GRAPH[@]}"
        run moe-base-paired "$BASE" "$MOE" --layout paired --microbatch 16 "${GRAPH[@]}"
        ;;
    replay)
        run dense-replay-reference "$BASE" "$DENSE" --cap 512 --microbatch 16 --compact 0 --replay
        run dense-replay-optimized "$BASE" "$DENSE" --cap 512 --microbatch 16 --compact 0 --replay "${GRAPH[@]}"
        run moe-replay-reference "$BASE" "$MOE" --cap 512 --microbatch 8 --compact 0 --replay
        run moe-replay-optimized "$BASE" "$MOE" --cap 512 --microbatch 8 --compact 0 --replay "${GRAPH[@]}"
        ;;
    profile)
        run dense-reference-profile "$BASE" "$DENSE" --cap 32 --microbatch 16 --repeats 1 --profile
        run moe-reference-profile "$BASE" "$MOE" --cap 32 --microbatch 8 --repeats 1 --profile
        run dense-profile "$BASE" "$DENSE" --cap 32 --microbatch 16 --repeats 1 --profile --stop-diagnostics "${GRAPH[@]}"
        run moe-profile "$BASE" "$MOE" --cap 32 --microbatch 8 --repeats 1 --profile --stop-diagnostics "${GRAPH[@]}"
        run dense-paired-profile "$BASE" "$DENSE" --cap 32 --layout paired --microbatch 32 --repeats 1 --profile "${GRAPH[@]}"
        run moe-paired-profile "$BASE" "$MOE" --cap 32 --layout paired --microbatch 16 --repeats 1 --profile "${GRAPH[@]}"
        run dense-profile-late "$BASE" "$DENSE" --microbatch 16 --repeats 1 --profile \
            --warmup-cap 32 --profile-start 4096 --stop-diagnostics "${GRAPH[@]}"
        run moe-profile-late "$BASE" "$MOE" --microbatch 8 --repeats 1 --profile \
            --warmup-cap 32 --profile-start 4096 --stop-diagnostics "${GRAPH[@]}"
        ;;
    singles)
        run student-only "$BASE" "$DENSE" --alpha 0 --microbatch 16 --repeats 1 "${GRAPH[@]}"
        run dense-teacher-only "$BASE" "$DENSE" --alpha 1 --microbatch 16 --repeats 1 "${GRAPH[@]}"
        run moe-teacher-only "$BASE" "$MOE" --alpha 1 --microbatch 16 --repeats 1 "${GRAPH[@]}"
        ;;
    *) echo "Unknown performance block: $BLOCK" >&2; exit 2 ;;
esac
exit
}
