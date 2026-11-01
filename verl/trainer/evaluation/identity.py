# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One content identity for rollout weights and reusable evaluation banks."""

import dataclasses

from verl.experimental.rollout_policy.identity import checkpoint_identity

from .artifacts import file_sha256


def model_identity(path):
    """Bind complete weights, configuration and tokenizer/template inputs."""
    return "checkpoint-v2:" + checkpoint_identity(path)


def bootstrap_contract(options, *, student, teacher):
    contract = {
        "student": student,
        "teacher": teacher,
        "datasets": [{"spec": dataclasses.asdict(d), "sha256": file_sha256(d.path)} for d in options.datasets],
        "protocols": {k: v.metadata() for k, v in options.protocols.items()},
    }
    if options.source_enabled:
        contract["source_sha256"] = file_sha256(options.source_file)
    return contract


def active_bootstrap_contract(stored, *, source_enabled):
    """An existing source bank may remain on disk when its evaluation is disabled."""
    result = dict(stored)
    if not source_enabled:
        result.pop("source_sha256", None)
    return result
