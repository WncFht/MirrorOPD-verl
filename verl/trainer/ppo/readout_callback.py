# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Optional driver readouts at completed optimizer/outer boundaries."""

from verl.utils.import_utils import load_class_from_fqn


def build_readout_callback(config):
    target = config.trainer.get("readout_callback_class")
    if not target:
        return None
    return load_class_from_fqn(target, "readout callback")(config=config)
