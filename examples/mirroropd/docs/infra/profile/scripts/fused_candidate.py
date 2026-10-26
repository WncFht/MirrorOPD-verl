# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compatibility import for the original profile candidate name.

The implementation lives in verl. Historical implementations are retained in
Git; this compatibility name always benchmarks the current production path.
"""

from verl.trainer.distillation.full_vocab_triton import fused_rows  # noqa: F401


class FusedPrecomputedKL:
    """Keep the archived scalar probe API without duplicating the production VJP."""

    @staticmethod
    def apply(hs, ws, ht, wt, token_chunk):
        from verl.trainer.distillation.full_vocab import full_reverse_kl

        return full_reverse_kl(hs, ws, ht, wt, backend="precompute_fused", token_chunk=token_chunk)
