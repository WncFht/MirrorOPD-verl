# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compatibility import for the original profile candidate name.

The implementation now lives in verl; archived experiment worktrees retain the
original standalone source. Existing benchmark commands remain reproducible.
"""

from verl.trainer.distillation.full_vocab_triton import (  # noqa: F401
    _gradient,
    _partials,
    _rows,
    fused_rows,
)


class FusedPrecomputedKL:
    """Keep the archived scalar probe API without duplicating the production VJP."""

    @staticmethod
    def apply(hs, ws, ht, wt, token_chunk):
        from verl.trainer.distillation.full_vocab import full_reverse_kl

        return full_reverse_kl(hs, ws, ht, wt, backend="precompute_fused", token_chunk=token_chunk)
