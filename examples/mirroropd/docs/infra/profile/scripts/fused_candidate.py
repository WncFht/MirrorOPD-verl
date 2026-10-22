# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compatibility import for the original profile candidate name.

The implementation now lives in verl; archived experiment worktrees retain the
original standalone source. Existing benchmark commands remain reproducible.
"""

from verl.trainer.distillation.full_vocab_triton import (  # noqa: F401
    FusedPrecomputedKL,
    _gradient,
    _partials,
    _rows,
    fused_rows,
)
