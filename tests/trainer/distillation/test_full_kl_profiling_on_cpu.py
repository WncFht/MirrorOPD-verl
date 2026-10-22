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


def test_semantic_attribution_handles_triton_driver_launch(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    analyze = load("semantic_trace")
    events = [
        dict(
            ph="X",
            cat="user_annotation",
            name="full_vocab/head_loss",
            pid=1,
            tid=1,
            ts=0,
            dur=100,
            args={"External id": 1},
        ),
        dict(
            ph="X",
            cat="user_annotation",
            name="full_vocab/fused_kl",
            pid=1,
            tid=1,
            ts=10,
            dur=20,
            args={"External id": 2},
        ),
        dict(ph="X", cat="cpu_op", name="aten::sum", pid=1, tid=1, ts=11, dur=1, args={"External id": 3}),
        dict(ph="X", cat="cuda_driver", name="cuLaunchKernelEx", pid=1, tid=1, ts=12, dur=1, args={"correlation": 77}),
        dict(cat="kernel", name="_partials", ts=200, dur=1000, args={"External id": 1, "correlation": 77}),
        dict(cat="kernel", name="reduce", ts=1200, dur=2000, args={"External id": 3}),
        dict(cat="kernel", name="unmapped", ts=3200, dur=3000, args={}),
    ]
    result = analyze.attribute(events)
    assert result["semantic_kernel_sum_ms"] == {"full_vocab/fused_kl": 3.0, "unattributed": 3.0}
    assert load("analyze").category("nvjet_tss_192x192_64x3_2x1_v_badd_coopB_NTN") == "GEMM"
    assert load("analyze").category("void flash::flash_fwd_kernel<cutlass::bfloat16_t>") == "Attention"


def test_allocator_requested_bytes_and_segment_replay():
    module = load("memory_timeline")
    snapshot = {
        "segments": [
            {
                "device": 0,
                "total_size": 1024,
                "blocks": [
                    {"state": "active_allocated", "size": 512, "requested_size": 20},
                    {"state": "inactive", "size": 512, "requested_size": 0},
                ],
            }
        ],
        "device_traces": [
            [
                {"action": "segment_alloc", "size": 2048, "time_us": 0},
                {"action": "alloc", "size": 100, "time_us": 100},
                {"action": "free_requested", "size": 100, "time_us": 6000},
                {"action": "free_completed", "size": 100, "time_us": 7000},
                {"action": "segment_free", "size": 2048, "time_us": 11000},
            ]
        ],
    }
    rows = module.reconstruct(snapshot)
    assert max(r["requested_live_gib"] for r in rows) == 120 / 2**30
    assert max(r["reserved_gib"] for r in rows) == 3072 / 2**30
    assert rows[-1]["requested_live_gib"] == 20 / 2**30
    assert rows[-1]["reserved_gib"] == 1024 / 2**30
