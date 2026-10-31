# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Resolve application references before constructing immutable rollout plans."""

from pathlib import Path

from .plan import ModelBinding


def local_model_path(value):
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Rollout requires a local checkpoint directory: {path}")
    return str(path)


def resolve_reference(reference, learner_path, teacher_path=None, *, resolve_path=local_model_path):
    if reference.kind in {"current", "initial"}:
        path = learner_path
    elif reference.kind == "teacher":
        if teacher_path is None:
            raise ValueError("teacher requires exactly one supervision teacher; use checkpoint explicitly")
        path = teacher_path
    else:
        path = reference.ref
    return ModelBinding("current" if reference.kind == "current" else "checkpoint", resolve_path(path))


def resolve_bindings(policy, learner_path, teacher_path=None, *, resolve_path=local_model_path):
    locations = {}

    def once(path):
        if path not in locations:
            locations[path] = resolve_path(path)
        return locations[path]

    return {
        slot: resolve_reference(ref, learner_path, teacher_path, resolve_path=once)
        for slot, ref in policy.references.items()
    }


def reachable_sources(policy, bindings):
    """Representative bindings needed by a schedule after identity folding."""
    if policy.mode == "single":
        return ("model",)
    rule = policy.right_weight
    if (rule.kind == "constant" or rule.start == rule.end) and rule.start in (0, 1):
        return ("left" if rule.start == 0 else "right",)
    if bindings["left"].same_model(bindings["right"]):
        return ("left",)
    return ("left", "right")


def requires_token_generator(policy, bindings):
    if policy.mode != "token_mixture" or len(reachable_sources(policy, bindings)) == 1:
        return False
    rule = policy.right_weight
    # Integer clocks jump directly between endpoints for a one-step schedule.
    return not (rule.kind == "linear" and rule.steps == 1 and rule.start in (0, 1) and rule.end in (0, 1))


def load_model_metadata(paths):
    from transformers import AutoConfig, AutoTokenizer

    return {
        path: (
            AutoConfig.from_pretrained(path, local_files_only=True),
            AutoTokenizer.from_pretrained(path, local_files_only=True),
        )
        for path in dict.fromkeys(paths)
    }
