# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Optional driver readouts at completed optimizer/outer boundaries."""

from collections.abc import Mapping

from omegaconf import OmegaConf

from verl.utils.import_utils import load_class_from_fqn


class ReadoutCallback:
    """Completed-boundary lifecycle; subclasses declare validation ownership."""

    handles_validation = False
    uses_validation_dataset = True
    requires_full_batch = False
    owns_final_weight_sync = False

    @staticmethod
    def preflight_generation_protocols(options):
        """Declare standalone generation protocols; native validation uses trainer budgets."""
        return {}

    def on_readout(self, trainer, outer):
        return {}

    def on_before_weight_sync(self, trainer, batch, outer):
        return {}

    def on_update_records(self, events, role):
        pass

    def validate(self, trainer):
        raise NotImplementedError("This readout does not own validation")


def readout_definition(config):
    """Resolve the single construction boundary without creating a readout."""
    retired = {
        "readout_callback_class",
        "rkl_readout",
        "source_history_readout",
        "hf_reload_readout",
        "rkl_probe",
        "scientific_evaluation",
    } & set(config.trainer)
    if retired:
        raise ValueError(f"Retired evaluation fields {sorted(retired)}; use trainer.evaluation._target_ and options")
    value = config.trainer.get("evaluation")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("trainer.evaluation requires a mapping or null")
    options = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else dict(value)
    target = options.pop("_target_", None)
    if not isinstance(target, str) or not target:
        raise ValueError("trainer.evaluation requires an explicit _target_")
    callback = load_class_from_fqn(target, "evaluation callback")
    if not isinstance(callback, type) or not issubclass(callback, ReadoutCallback):
        raise TypeError("Evaluation targets must implement the ReadoutCallback lifecycle")
    return callback, options


def readout_options(config, expected):
    """Standalone tools require the same declared protocol as the trainer."""
    definition = readout_definition(config)
    if definition is None or not issubclass(definition[0], expected):
        raise ValueError(f"This tool requires evaluation target {expected.__name__}")
    return definition[1]


def build_readout_callback(config):
    definition = readout_definition(config)
    if definition is None:
        return None
    callback, options = definition
    return callback(config=config, options=options)
