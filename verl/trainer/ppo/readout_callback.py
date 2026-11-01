# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Optional driver readouts at completed optimizer/outer boundaries."""

from verl.utils.import_utils import load_class_from_fqn


class ReadoutCallback:
    """Completed-boundary lifecycle; subclasses declare validation ownership."""

    handles_validation = False

    def on_readout(self, trainer, outer):
        return {}

    def on_before_weight_sync(self, trainer, batch, outer):
        return {}

    def on_update_records(self, events, role):
        pass

    def validate(self, trainer):
        raise NotImplementedError("This readout does not own validation")


def build_readout_callback(config):
    if config.trainer.get("rkl_readout"):
        raise ValueError("Migrate rkl_readout to source_history_readout and SourceHistoryReadout")
    if config.trainer.get("evaluation"):
        if config.trainer.get("readout_callback_class"):
            raise ValueError("Choose unified evaluation or an explicit readout callback, not both")
        from verl.trainer.evaluation.coordinator import EvaluationCoordinator

        return EvaluationCoordinator(config=config)
    target = config.trainer.get("readout_callback_class")
    if not target:
        return None
    callback = load_class_from_fqn(target, "readout callback")
    if not issubclass(callback, ReadoutCallback):
        raise TypeError("Readout callbacks must implement the ReadoutCallback lifecycle")
    return callback(config=config)
