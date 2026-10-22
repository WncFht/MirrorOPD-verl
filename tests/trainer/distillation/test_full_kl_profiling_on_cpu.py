# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from verl.trainer.distillation.profiling import full_kl_range

SCRIPTS = Path(__file__).resolve().parents[3] / "examples/mirroropd/docs/infra/profile/scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_range_unwinds_nvtx_on_failure(monkeypatch):
    calls = []
    monkeypatch.setenv("VERL_FULL_KL_NVTX", "1")
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda name: calls.append(name))
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: calls.append("pop"))
    with pytest.raises(RuntimeError):
        with full_kl_range("probe"):
            raise RuntimeError("expected")
    assert calls == ["full_vocab/probe", "pop"]


def test_trace_overlap_uses_interval_union(tmp_path):
    analyze = load("analyze")
    path = tmp_path / "trace.json"
    # Two overlapping compute kernels must not be double-counted as occupied time.
    events = [
        dict(cat="kernel", name=n, ts=t, dur=d, args={"device": 0})
        for n, t, d in [("gemm", 0, 10000), ("gemm", 5000, 10000), ("ncclAllGather", 10000, 10000)]
    ]
    path.write_text(json.dumps({"traceEvents": events}))
    summary, _ = analyze.trace_stats(path)
    assert summary["kernel_sum_ms_by_category"]["GEMM"] == 20
    assert summary["devices"]["0"] == {
        "compute_union_ms": 15,
        "nccl_union_ms": 10,
        "kernel_union_ms": 20,
        "overlap_ms": 5,
        "nccl_without_compute_ms": 5,
        "first_to_last_kernel_ms": 20,
    }


def test_select_chunk_ignores_oom_and_wrong_length():
    select = load("run_study").select_chunk
    result = {
        "rows": [
            {"backend": "precompute", "tokens": 32768, "chunk": 1024, "status": "oom"},
            {"backend": "precompute", "tokens": 4096, "chunk": 128, "wall_ms_median": 1},
            {"backend": "precompute", "tokens": 32768, "chunk": 512, "wall_ms_median": 3},
            {"backend": "precompute", "tokens": 32768, "chunk": 256, "wall_ms_median": 4},
        ]
    }
    assert select(result, "precompute") == 512
    assert select(result, "fused_candidate") is None
