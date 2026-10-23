# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Semantic full-KL ranges; optional NVTX without synchronizing CUDA."""

import os
from contextlib import contextmanager, nullcontext

import torch

from verl.utils.profiler.timeline import span


@contextmanager
def full_kl_range(name):
    """NVTX is opt-in for nsys; record_function works with the existing torch profiler."""
    nvtx = os.environ.get("VERL_FULL_KL_NVTX") == "1"
    coarse = name in {
        "teacher_forward",
        "student_forward",
        "backbone",
        "head_loss",
        "micro_batch_h2d",
        "teacher_hidden_d2h",
        "teacher_head_get",
        "hidden_read",
        "hidden_write",
        "hidden_read_wait",
        "hidden_write_wait",
        "hidden_write_flush",
    }
    host = span("full_vocab/" + name, role="fsdp") if coarse else nullcontext()
    with torch.profiler.record_function("full_vocab/" + name), host:
        if nvtx:
            torch.cuda.nvtx.range_push("full_vocab/" + name)
        try:
            yield
        finally:
            if nvtx:
                torch.cuda.nvtx.range_pop()
