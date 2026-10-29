# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Report the loaded actor version, never the collection counter."""


def output_metadata(request_id, phase_token, loaded_version, alpha, teacher_path):
    has_current = alpha < 1 and loaded_version is not None and isinstance(loaded_version[0], int)
    return {
        "fusion_request_id": request_id,
        "fusion_backend": "vllm_native",
        "fusion_phase_token": phase_token,
        "fusion_loaded_version": loaded_version,
        "global_steps": loaded_version[0] if has_current else None,
        "actor_version": loaded_version if has_current else None,
        "student_version": loaded_version if alpha < 1 else None,
        "teacher_path": teacher_path,
        "teacher_weight": alpha,
    }
