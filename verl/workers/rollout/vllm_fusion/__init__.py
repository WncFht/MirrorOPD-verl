# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Opt-in native vLLM dual-model execution."""


def register_models():
    """Register lazily in every vLLM process without changing native models."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "MirrorOPDFusionForCausalLM",
        "verl.workers.rollout.vllm_fusion.model:MirrorOPDFusionForCausalLM",
    )
