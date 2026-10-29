#!/usr/bin/env bash
# Formal single-GPU workload; overrides change concurrency, never the declared cap.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../.." && pwd)
WORKSPACE=$(dirname -- "$REPO")
PAIR=${1:?dense or moe}
PREFIX=${2:?unique output directory name}
MODE=${3:?eager or graph}
case "$PAIR" in
    dense)
        TEACHER="$WORKSPACE/model/Qwen3-8B"
        SEQS=${FUSION_MAX_NUM_SEQS:-16}
        KV=${FUSION_KV_GIB:-40}
        ;;
    moe)
        TEACHER="$WORKSPACE/model/Qwen3-30B-A3B-Instruct-2507"
        SEQS=${FUSION_MAX_NUM_SEQS:-4}
        KV=${FUSION_KV_GIB:-10}
        ;;
    *) exit 2 ;;
esac
EXTRA=()
case "$MODE" in eager) ;; graph) EXTRA+=(--graph) ;; *) exit 2 ;; esac
bash "$HERE/run_fusion_local_fsdp2.sh" "$WORKSPACE/envs/mirroropd-fusion/bin/python" \
    "$HERE/probe_vllm.py" \
    --student "$WORKSPACE/model/Qwen3-1.7B-Base" --teacher "$TEACHER" \
    --data "$WORKSPACE/data/math/DAPO/train.parquet" \
    --out "$WORKSPACE/runs/vllm-fusion-20261029/$PREFIX" \
    --batch 64 --cap 7168 --max-model-len 11264 --alphas 0.5 --repeats 3 \
    --max-num-seqs "$SEQS" --kv-gib "$KV" "${EXTRA[@]}"
