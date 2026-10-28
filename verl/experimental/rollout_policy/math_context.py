# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Scope inference arithmetic settings without changing the learner's execution."""

from contextlib import contextmanager, nullcontext

import torch


@contextmanager
def inference_math(*, cuda=True, allow_cudnn=False):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    context = (
        nullcontext()
        if allow_cudnn
        else sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        )
    )
    matmul = torch.backends.cuda.matmul
    previous = None
    if cuda and hasattr(matmul, "allow_bf16_reduced_precision_reduction_split_k"):
        previous = (
            torch.backends.cuda.preferred_blas_library(),
            matmul.allow_bf16_reduced_precision_reduction,
            matmul.allow_bf16_reduced_precision_reduction_split_k,
            matmul.allow_tf32,
        )
    try:
        if previous is not None:
            # Split-K in the 4096-wide Qwen3 teacher was shape-dependent even
            # with reduced-precision reduction disabled. Torch >=2.10 exposes
            # the independent split-K control, which requires cuBLASLt.
            torch.backends.cuda.preferred_blas_library("cublaslt")
            matmul.allow_bf16_reduced_precision_reduction = (False, False)
            matmul.allow_tf32 = False
        with context:
            yield
    finally:
        if previous is not None:
            backend, reduced, split_k, tf32 = previous
            matmul.allow_bf16_reduced_precision_reduction = (reduced, split_k)
            matmul.allow_tf32 = tf32
            torch.backends.cuda.preferred_blas_library(backend)
