# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Check Triton reduction algebra without taking a GPU; CUDA validation is separate."""

import os
import subprocess
import sys

import pytest
import torch


def _check(vocab, scale):
    from verl.trainer.distillation.full_vocab_triton import fused_rows

    torch.manual_seed(42)
    a, b = [torch.randn(3, vocab) * scale for _ in range(2)]
    kl, grad = fused_rows(a, b, torch.float32)
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
    # Triton 3.6's CPU interpreter does not implement CUDA libdevice calls.
    # Emulate only those primitives with NumPy; real CUDA tests cover libdevice.
    import numpy as np
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
    from triton.runtime.interpreter import TensorHandle

    def unary(fn):
        def evaluate(x):
            return tl.tensor(TensorHandle(fn(x.handle.data), x.handle.dtype), x.type)

        return evaluate

    libdevice.log1p = unary(np.log1p)
    libdevice.expm1 = unary(np.expm1)
    for vocab in (31, 4096, 8193, 151936):
        for scale in (1.0, 20.0):
            _check(vocab, scale)
