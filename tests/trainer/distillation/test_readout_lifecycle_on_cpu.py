# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""All readouts implement one lifecycle; missing methods are not runtime capabilities."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo import readout_callback


class ExampleReadout(readout_callback.ReadoutCallback):
    def __init__(self, *, config):
        self.config = config


def test_optional_hooks_and_validation_ownership_are_explicit(monkeypatch):
    config = OmegaConf.create({"trainer": {"readout_callback_class": "example.Callback"}})
    monkeypatch.setattr(readout_callback, "load_class_from_fqn", lambda *args: ExampleReadout)
    callback = readout_callback.build_readout_callback(config)
    assert callback.on_before_weight_sync(None, None, 1) == {}
    assert callback.on_readout(None, 1) == {}
    assert callback.on_update_records([], "actor") is None
    assert not callback.handles_validation
    with pytest.raises(NotImplementedError):
        callback.validate(None)


def test_partial_callbacks_and_retired_configuration_fail_at_construction(monkeypatch):
    monkeypatch.setattr(readout_callback, "load_class_from_fqn", lambda *args: SimpleNamespace)
    with pytest.raises(TypeError, match="lifecycle"):
        readout_callback.build_readout_callback(
            OmegaConf.create({"trainer": {"readout_callback_class": "old.Callback"}})
        )
    with pytest.raises(ValueError, match="Migrate"):
        readout_callback.build_readout_callback(OmegaConf.create({"trainer": {"rkl_readout": {"output": "old"}}}))


@pytest.mark.parametrize(
    "module,name,owns_validation",
    [
        ("verl.trainer.evaluation.coordinator", "EvaluationCoordinator", True),
        ("verl.trainer.evaluation.source_history", "SourceHistoryReadout", True),
        ("verl.trainer.evaluation.reload", "HFReloadReadout", True),
        ("examples.mirroropd.benchmarks.profile.native_evaluation", "NativeRKLProbe", True),
        ("examples.mirroropd.research.studies.prefix_fusion.callback", "ScientificReadout", False),
    ],
)
def test_all_live_readout_consumers_declare_the_lifecycle(module, name, owns_validation):
    import importlib

    callback = getattr(importlib.import_module(module), name)
    assert issubclass(callback, readout_callback.ReadoutCallback)
    assert callback.handles_validation is owns_validation
    if owns_validation:
        assert callback.validate is not readout_callback.ReadoutCallback.validate
