# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Trainer hooks select their rollout delegate once and preserve default transport."""

from types import SimpleNamespace
from unittest.mock import Mock

from verl.experimental.rollout_policy.plan import RolloutPolicyConfig
from verl.experimental.rollout_policy.service import DefaultRolloutService, PolicyRolloutService, create_rollout_service


def test_default_service_preserves_sync_and_transport_order():
    events = []
    trainer = SimpleNamespace(
        rollout_policy=RolloutPolicyConfig(),
        global_steps=3,
        timing_raw={},
        checkpoint_manager=SimpleNamespace(
            update_weights=lambda version: events.append(("sync", version)),
            sleep_replicas=lambda: events.append(("sleep",)),
        ),
    )
    service = create_rollout_service(trainer, lambda batch: events.append(("submit", batch)))
    assert isinstance(service, DefaultRolloutService)
    service.initialize()
    service.submit("batch")
    service.after_sample()
    service.after_update()
    service.validation_begin()
    service.validation_end()
    assert events == [("sync", 3), ("submit", "batch"), ("sleep",), ("sync", 3)]


def test_policy_service_owns_collection_and_validation(monkeypatch):
    from verl.experimental.rollout_policy import runtime
    from verl.experimental.rollout_policy import service as module

    runner = SimpleNamespace(collect=Mock(return_value={"count": 2}), sync_validation=Mock(), sleep_validation=Mock())
    monkeypatch.setattr(runtime, "RolloutRuntime", lambda trainer: runner)
    publish = Mock()
    monkeypatch.setattr(module.tq, "kv_batch_put", publish)
    trainer = SimpleNamespace(
        rollout_policy=RolloutPolicyConfig.parse({"model": {"kind": "initial"}}),
        global_steps=3,
        _stamp_rollout_submission=Mock(),
    )
    default_submit = Mock()
    service = create_rollout_service(trainer, default_submit)
    assert isinstance(service, PolicyRolloutService)
    service.submit({"uid": ["a", "b"], "index": [1, 2]})
    assert service.metrics == {"count": 2}
    service.validation_begin()
    service.validation_end()
    runner.sync_validation.assert_called_once()
    runner.sleep_validation.assert_called_once()
    default_submit.assert_not_called()
    assert publish.call_args.kwargs["keys"] == ["a", "b"]
