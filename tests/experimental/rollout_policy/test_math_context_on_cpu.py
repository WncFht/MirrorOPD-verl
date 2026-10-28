# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from verl.experimental.rollout_policy.math_context import inference_math


@pytest.mark.parametrize("fail", [False, True])
def test_inference_settings_restore_training_backend(monkeypatch, fail):
    class Matmul:
        reduced = True
        split_k = True
        allow_tf32 = True

        @property
        def allow_bf16_reduced_precision_reduction(self):
            return self.reduced

        @allow_bf16_reduced_precision_reduction.setter
        def allow_bf16_reduced_precision_reduction(self, value):
            self.reduced, self.split_k = value

        @property
        def allow_bf16_reduced_precision_reduction_split_k(self):
            return self.split_k

    state = {"backend": "cublas"}

    def preferred(value=None):
        if value is not None:
            state["backend"] = value
        return state["backend"]

    matmul = Matmul()
    monkeypatch.setattr(torch.backends.cuda, "matmul", matmul)
    monkeypatch.setattr(torch.backends.cuda, "preferred_blas_library", preferred)

    def run():
        with inference_math(cuda=True):
            assert state["backend"] == "cublaslt"
            assert not matmul.reduced and not matmul.split_k
            assert not matmul.allow_tf32
            assert not torch.backends.cuda.cudnn_sdp_enabled()
            if fail:
                raise RuntimeError("decode failed")

    before = torch.backends.cuda.cudnn_sdp_enabled()
    if fail:
        with pytest.raises(RuntimeError, match="decode failed"):
            run()
    else:
        run()
    assert state["backend"] == "cublas"
    assert matmul.reduced and matmul.split_k
    assert matmul.allow_tf32
    assert torch.backends.cuda.cudnn_sdp_enabled() == before
