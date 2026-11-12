# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Verify functional spectrum edits and paired bank aggregation independently."""

import pytest
import torch

from examples.mirroropd.research.studies.absorption_dynamics.spectrum_bank_inputs import validate_rows
from examples.mirroropd.research.studies.absorption_dynamics.spectrum_bank_worker import paired_summary, restore_sigma0
from verl.trainer.evaluation.reporting import evaluation_metrics


@pytest.mark.parametrize("shape", [(9, 5), (5, 9), (7, 7)])
def test_gram_restoration_matches_svd_without_truncation(shape):
    from examples.mirroropd.research.studies.absorption_dynamics.spectrum_bank_acceleration import gram_restore

    torch.manual_seed(55)
    w = torch.randn(shape, dtype=torch.float64)
    s0 = torch.linalg.svdvals(w + 0.02 * torch.randn_like(w))
    actual, audit = gram_restore(w, s0, fallback=restore_sigma0)
    expected, _ = restore_sigma0(w, s0)
    assert audit["method"] == "gram-fp64"
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(torch.linalg.svdvals(actual.double()), s0, atol=2e-6, rtol=1e-6)


def test_gram_restoration_falls_back_for_near_null_directions():
    from examples.mirroropd.research.studies.absorption_dynamics.spectrum_bank_acceleration import gram_restore

    w = torch.diag(torch.tensor([1.0, 0.1, 1e-12], dtype=torch.float64))
    s0 = torch.tensor([1.1, 0.12, 1e-10], dtype=torch.float64)
    actual, audit = gram_restore(w, s0, fallback=restore_sigma0)
    expected, _ = restore_sigma0(w, s0)
    assert audit["method"] == "svd-fallback"
    torch.testing.assert_close(actual, expected)


def test_reference_device_cache_is_bounded_and_validates_row_identity(tmp_path):
    from examples.mirroropd.research.studies.absorption_dynamics.spectrum_bank_acceleration import ReferenceCache
    from verl.trainer.evaluation.artifacts import identity

    torch.save(torch.zeros(2, 2), tmp_path / "head.pt")
    rows = [dict(id=str(i), response_ids=[i]) for i in range(3)]
    for row in rows:
        torch.save(dict(row_identity=identity(row), hidden=torch.ones(2, 2)), tmp_path / f"{identity(row['id'])}.pt")
    cache = ReferenceCache(tmp_path, "cpu", max_bytes=16)
    cache.get(rows[0])
    cache.get(rows[0])
    cache.get(rows[1])
    assert cache.stats() == dict(hits=1, misses=2, resident_bytes=16, limit_bytes=16)
    assert len(cache.entries) == 1
    with pytest.raises(ValueError, match="row changed"):
        cache.get(dict(rows[2], response_ids=[999]))


@pytest.mark.parametrize("shape", [(9, 5), (5, 9)])
def test_sigma0_full_weight_edit_preserves_target_frames(shape):
    torch.manual_seed(49)
    base = torch.randn(shape, dtype=torch.float64)
    target = base + 0.03 * torch.randn_like(base)
    u, s, vh = torch.linalg.svd(target, full_matrices=False)
    s0 = torch.linalg.svdvals(base)
    edited, receipt = restore_sigma0(target, s0)
    torch.testing.assert_close(edited.double(), (u * s0) @ vh, atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(torch.linalg.svdvals(edited.double()), s0, atol=2e-6, rtol=1e-6)
    assert receipt["reconstruction_relative_error"] < 1e-12
    unchanged, _ = restore_sigma0(target, s)
    assert torch.equal(unchanged, target.float())


def test_paired_delta_uses_actual_token_count_and_equal_domain_mean():
    before = []
    for domain, count, tokens, value in (("math", 2, 7168, 0.3), ("code", 1, 302, 0.6), ("if", 1, 8, 0.9)):
        for i in range(count):
            before.append(
                dict(
                    id=f"{domain}:{i}",
                    bank="public_initial",
                    domain=domain,
                    tokens=tokens,
                    rkl_sum=tokens * value,
                    rkl_position_sum=[tokens * value] + [0.0] * 9,
                    rkl_position_count=[tokens] + [0] * 9,
                )
            )
    after = [
        dict(r, rkl_sum=r["rkl_sum"] * 1.1, rkl_position_sum=[v * 1.1 for v in r["rkl_position_sum"]]) for r in before
    ]
    a = dict(rows=before, metrics=evaluation_metrics(before, baselines={}))
    b = dict(rows=after, metrics=evaluation_metrics(after, baselines={}))
    result = paired_summary(a, b, before)
    key = "eval/rkl/public_initial/all/question_mean"
    assert result["original"][key] == pytest.approx(0.6)
    assert result["delta"][key] == pytest.approx(0.06)
    assert result["training_comparison"]["exact_native_gate"]
    after[0]["tokens"] -= 1
    with pytest.raises(ValueError, match="row masks"):
        paired_summary(a, b, before)


def test_fixed_bank_rejects_duplicate_or_changed_masks():
    rows = []
    for bank in ("public_initial", "public_teacher"):
        for domain, n in (("math", 128), ("code", 64), ("if", 64)):
            for i in range(n):
                rows.append(
                    dict(
                        id=f"{bank}:{domain}:{i}",
                        bank=bank,
                        domain=domain,
                        prompt_ids=[1],
                        response_ids=[2, 3],
                        response_mask=[True, False],
                        valid_tokens=1,
                    )
                )
    points = [dict(training_rows=[dict(id=r["id"], tokens=1) for r in rows])]
    assert validate_rows(rows, points)
    rows[0]["response_mask"][1] = True
    with pytest.raises(ValueError, match="mask/cap"):
        validate_rows(rows, points)


def test_functional_collector_rejects_pilot(tmp_path):
    import json

    from examples.mirroropd.research.studies.absorption_dynamics.spectrum_bank_report import collect

    source = tmp_path / "pilot"
    source.mkdir()
    (source / "protocol.json").write_text(json.dumps(dict(smoke=True, pairs=["E94"])))
    with pytest.raises(ValueError, match="Smoke rows"):
        collect([source], tmp_path / "data")


def test_hidden_mask_keeps_original_prediction_positions():
    from types import SimpleNamespace

    from examples.mirroropd.research.studies.absorption_dynamics.spectrum_bank_worker import hidden

    class FakeModel:
        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(1))

        def get_decoder(self):
            def decode(input_ids, use_cache):
                assert not use_cache
                return SimpleNamespace(
                    last_hidden_state=torch.arange(input_ids.shape[1], dtype=torch.float32)[None, :, None]
                )

            return decode

    row = dict(prompt_ids=[9, 8, 7], response_ids=[1, 2, 3, 4], response_mask=[True, False, True, False])
    actual = hidden(FakeModel(), row)
    assert actual[:, 0].tolist() == [2, 4]
