# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Semantic full-KL ranges; optional NVTX without synchronizing CUDA."""

import os
from contextlib import contextmanager

import torch


@contextmanager
def full_kl_range(name):
    """NVTX is opt-in for nsys; record_function works with the existing torch profiler."""
    nvtx = os.environ.get("VERL_FULL_KL_NVTX") == "1"
    with torch.profiler.record_function("full_vocab/" + name):
        if nvtx:
            torch.cuda.nvtx.range_push("full_vocab/" + name)
        try:
            yield
        finally:
            if nvtx:
                torch.cuda.nvtx.range_pop()
