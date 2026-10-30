# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Long diagnostics cannot claim a complete training bank or a different S0 prefix."""

from types import SimpleNamespace

import pytest

from examples.mirroropd.research.core.io import dump
from examples.mirroropd.research.studies.fixed_transfer.data import bank_shard_receipt
from examples.mirroropd.research.studies.fixed_transfer.long_panel import verify_prefixes


def test_partial_bank_receipts_are_distinct_from_full_bank(tmp_path):
    study = SimpleNamespace(directory=tmp_path, cache_directory=tmp_path / "cache")
    dump(tmp_path / "prepared/complete.json", {"bank_key": "fixed"})
    full, a = bank_shard_receipt(study, 0)
    panel, b = bank_shard_receipt(study, 0, groups=["M02", "M01"])
    assert full != panel and a != b and "groups" not in a
    assert b["groups"] == ("M01", "M02")
    assert bank_shard_receipt(study, 0, groups=["M01", "M02"])[0] == panel
    with pytest.raises(ValueError, match="empty"):
        bank_shard_receipt(study, 0, groups=[])


def test_long_trajectory_must_preserve_prompt_and_original_response():
    old = [dict(id=1, prompt_ids=[7], response_ids=[1, 2]), dict(id=2, prompt_ids=[8], response_ids=[3])]
    extended = [dict(old[0], response_ids=[1, 2, 4]), dict(old[1])]
    assert verify_prefixes(old, extended)["matched"] == 2
    extended[0]["response_ids"] = [1, 9, 4]
    assert verify_prefixes(old, extended)["mismatched_ids"] == [1]
    extended[0]["response_ids"] = [1]
    assert verify_prefixes(old, extended)["mismatched_ids"] == [1]
