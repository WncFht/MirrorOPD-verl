# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Report the loaded actor version, never the collection counter."""


def output_metadata(
    request_id,
    phase_token,
    loaded_version,
    alpha,
    teacher_path,
    fusion_space="logit",
    *,
    has_current,
    reverse_components=False,
):
    primary_used = alpha > 0 if reverse_components else alpha < 1
    has_current = has_current and primary_used
    return {
        "fusion_request_id": request_id,
        "fusion_backend": "vllm_native",
        "fusion_phase_token": phase_token,
        "fusion_loaded_version": loaded_version,
        "global_steps": loaded_version[0] if has_current else None,
        "actor_version": loaded_version if has_current else None,
        "student_version": loaded_version if primary_used else None,
        "teacher_path": teacher_path,
        "right_weight": alpha,
        "reverse_components": reverse_components,
        "fusion_space": fusion_space,
    }
