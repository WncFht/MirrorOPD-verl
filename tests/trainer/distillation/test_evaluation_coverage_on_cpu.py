# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Event coverage is bound to selections, independently of scientific validity."""

import pytest

from verl.trainer.evaluation.coverage import rkl_coverage


def selected(key="q", bank="public_initial", **changes):
    return dict(id=key, bank=bank, trajectory_id="trajectory", valid_tokens=2, **changes)


def scored(reference, **changes):
    return (
        {k: v for k, v in reference.items() if k not in ("valid_tokens", "response_mask")}
        | dict(tokens=2, rkl_sum=1.0)
        | changes
    )


def test_same_trajectory_can_be_scored_in_multiple_banks_in_any_order():
    expected = [selected(), selected(bank="public_teacher")]
    coverage = rkl_coverage(expected, [scored(r) for r in reversed(expected)])
    assert coverage["scientific_complete"]
    assert coverage["expected_rows"] == coverage["observed_rows"] == coverage["valid_rows"] == 2
    assert set(coverage["banks"]) == {"public_initial", "public_teacher"}
    assert coverage == rkl_coverage(list(reversed(expected)), [scored(r) for r in expected])


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate_expected", "duplicate_observed", "wrong_bank"])
def test_incomplete_or_ambiguous_selections_fail_before_publication(change):
    expected = [selected()]
    observed = [scored(expected[0])]
    if change == "missing":
        observed = []
    elif change == "extra":
        observed.append(scored(selected(key="extra")))
    elif change == "duplicate_expected":
        expected *= 2
    elif change == "duplicate_observed":
        observed *= 2
    else:
        observed[0]["bank"] = "other"
    with pytest.raises(ValueError, match="coverage"):
        rkl_coverage(expected, observed)


@pytest.mark.parametrize("changes", [{"trajectory_id": "other"}, {"tokens": 1}])
def test_changed_trajectory_or_mask_fails(changes):
    reference = selected()
    with pytest.raises(ValueError, match="coverage"):
        rkl_coverage([reference], [scored(reference, **changes)])


@pytest.mark.parametrize("total", [float("nan"), float("inf"), -float("inf")])
def test_invalid_scores_remain_covered_but_scientifically_incomplete(total):
    reference = selected()
    coverage = rkl_coverage([reference], [scored(reference, rkl_sum=total)])
    assert coverage["observed_rows"] == coverage["invalid_rows"] == 1
    assert coverage["missing_rows"] == coverage["valid_rows"] == 0
    assert not coverage["scientific_complete"]
    assert not coverage["banks"][reference["bank"]]["scientific_complete"]


def test_zero_token_and_empty_events_do_not_claim_scientific_readouts():
    reference = dict(id="zero", bank="history", response_mask=[0, 0])
    coverage = rkl_coverage([reference], [scored(reference, tokens=0, rkl_sum=0)])
    assert coverage["invalid_rows"] == 1
    assert not coverage["scientific_complete"]
    empty = rkl_coverage([], [])
    assert empty["expected_rows"] == empty["observed_rows"] == 0
    assert not empty["scientific_complete"]


@pytest.mark.parametrize("before_sync", [False, True])
@pytest.mark.parametrize("failure", [None, "missing", "restore"])
def test_coordinator_publishes_coverage_only_after_required_restoration(tmp_path, monkeypatch, before_sync, failure):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from verl.trainer.evaluation import coordinator as module
    from verl.trainer.evaluation.artifacts import ArtifactStore
    from verl.trainer.evaluation.plan import EvaluationPlan

    coordinator = object.__new__(module.EvaluationCoordinator)
    reference = selected(source_id="source", domain="math")
    coordinator.pending_outer = 1
    coordinator.completed = set()
    coordinator.pending_events = {}
    coordinator.options = SimpleNamespace(final_export=False, score=None)
    coordinator.store = ArtifactStore.create(tmp_path / "run", kind="test", metadata={})
    coordinator.fixed = {"public_initial": [reference]}
    coordinator.history = []
    coordinator.baselines = {}
    coordinator.previous_history = []
    coordinator.hidden_size = 2
    coordinator.before_sync = before_sync
    coordinator._progress = Mock()
    service = SimpleNamespace(park=Mock(), ensure_current=Mock())
    if failure == "restore":
        service.ensure_current.side_effect = RuntimeError("restore failed")
    trainer = SimpleNamespace(total_training_steps=2, actor_optimizer_step=32, rollout_service=service)
    monkeypatch.setattr(module, "compile_plan", lambda *a, **k: EvaluationPlan(1, 32, ("public_initial",), False))
    monkeypatch.setattr(
        module, "score_native_rows", lambda *a, **k: ([] if failure == "missing" else [scored(reference)], {})
    )
    event_path = coordinator.store.root / "events/outer-001"
    if failure == "missing" or (failure == "restore" and not before_sync):
        with pytest.raises((ValueError, RuntimeError)):
            coordinator.validate(trainer)
        assert ArtifactStore(event_path).manifest()["status"] == "active"
        assert not coordinator.completed and not coordinator.pending_events
        return
    coordinator.validate(trainer)
    if before_sync:
        assert ArtifactStore(event_path).manifest()["status"] == "active"
        assert 1 in coordinator.pending_events
        service.ensure_current.assert_not_called()
        # The trainer calls on_readout only after the normal weight sync succeeds.
        coordinator.on_readout(trainer, 1)
    else:
        service.ensure_current.assert_called_once_with(1)
    metadata = ArtifactStore(event_path).manifest(complete=True)["metadata"]
    assert metadata["coverage"]["scientific_complete"]
    assert metadata["coverage"]["valid_rows"] == 1
    assert metadata["outer"] == 1 and metadata["optimizer_steps"] == 32
