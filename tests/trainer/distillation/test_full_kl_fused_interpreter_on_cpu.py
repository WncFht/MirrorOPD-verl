# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Check Triton reduction algebra without taking a GPU; CUDA validation is separate."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _check(vocab, scale):
    import triton

    path = Path(__file__).resolve().parents[3] / "examples/mirroropd/docs/infra/profile/scripts/fused_candidate.py"
    spec = importlib.util.spec_from_file_location("fused_interpreter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.manual_seed(42)
    n, parts = 3, triton.cdiv(vocab, 4096)
    a, b = [torch.randn(n, vocab) * scale for _ in range(2)]
    buffers = [torch.empty(n, parts) for _ in range(5)]
    lz, mu, kl = [torch.empty(n) for _ in range(3)]
    grad = torch.empty(n, vocab)
    mod._partials[(n, parts)](a, b, *buffers, vocab, parts, 4096)
    mod._rows[(n,)](*buffers, lz, mu, kl, parts, triton.next_power_of_2(parts))
    mod._gradient[(n, parts)](a, b, lz, mu, grad, vocab, 4096)
    lp, lt = a.log_softmax(-1), b.log_softmax(-1)
    expected = (lp.exp() * (lp - lt)).sum(-1)
    torch.testing.assert_close(kl, expected, atol=2e-4, rtol=2e-5)
    torch.testing.assert_close(grad, lp.exp() * (lp - lt - expected[:, None]), atol=2e-4, rtol=2e-4)


def test_split_vocab_reduction():
    pytest.importorskip("triton")
    # Interpreter mode must be set before Triton's standard library is imported.
    subprocess.run(
        [sys.executable, __file__],
        check=True,
        timeout=120,
        env=os.environ | {"TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    )


if __name__ == "__main__":
    for vocab in (31, 4096, 8193, 151936):
        for scale in (1.0, 20.0):
            _check(vocab, scale)
