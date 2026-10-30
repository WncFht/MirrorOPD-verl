# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Equal-budget curves pool prompts and retain the entire frozen training panel."""

import pytest

from examples.mirroropd.research.studies.fixed_transfer.plot_learning import panel_means


def test_fixed_panels_use_prompt_weighting_and_require_complete_support():
    groups = list("ABCDEFGHIJKLMNOPQRST") + ["M01", "M02", "M03", "M04", "Code01", "Code02", "IF01", "IF02"]
    rows = []
    for group in groups:
        value = 2 if group.startswith("M0") else 4 if group.startswith("Code") else 8 if group.startswith("IF") else 1
        for _ in range(16):
            rows.append(dict(id=len(rows), group=group, stats={"all": {"rkl": value * 8}}))
    result = panel_means(rows, 8)
    assert result["train_all"] == result["old_train_ad"] == 1
    assert result["held_all"] == 4
    assert result["held_math"] == 2 and result["held_code"] == 4 and result["held_if"] == 8
    with pytest.raises(ValueError, match="Incomplete"):
        panel_means(rows[:-1], 8)
    with pytest.raises(ValueError, match="Duplicate"):
        panel_means(rows + rows[:1], 8)
