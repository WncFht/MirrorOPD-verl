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
    return {
        "student": student,
        "teacher": teacher,
        "source_sha256": file_sha256(options.source_file),
        "datasets": [{"spec": dataclasses.asdict(d), "sha256": file_sha256(d.path)} for d in options.datasets],
        "protocols": {k: v.metadata() for k, v in options.protocols.items()},
    }
