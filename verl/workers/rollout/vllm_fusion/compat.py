# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Pinned vLLM 0.25.1 adapter for complete, independent child model configs."""

import copy


def validate_component_families(student, teacher):
    """Keep the tested dense learner families and native teacher combinations explicit."""
    if (student.model_type, teacher.model_type) not in {
        ("qwen2", "qwen2"),
        ("qwen3", "qwen3"),
        ("qwen3", "qwen3_moe"),
    }:
        raise ValueError("Native fusion requires Qwen2 dense pairs or a Qwen3 dense learner with dense/MoE teacher")


def decode_graph_config(max_num_seqs):
    """Capture native decode shapes while keeping prefill and model arithmetic eager."""
    sizes = []
    size = 1
    while size < max_num_seqs:
        sizes.append(size)
        size *= 2
    sizes.append(max_num_seqs)
    return {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": sizes}


def fusion_hf_overrides(teacher_path):
    """Expose teacher experts before vLLM initializes its model-parallel groups."""
    from transformers import AutoConfig

    teacher = AutoConfig.from_pretrained(teacher_path, local_files_only=True)
    return {
        "architectures": ["MirrorOPDFusionForCausalLM"],
        "fusion_teacher_path": teacher_path,
        "num_experts": getattr(teacher, "num_experts", 0),
    }


def component_config(parent, hf_config, checkpoint):
    """Replace both HF views before deriving architecture and model capabilities.

    vLLM 0.25.1 with_hf_config leaves hf_text_config pointing at the outer
    model. Qwen3MoE reads that view, so using the helper alone constructs the
    teacher from student dimensions. Execution/cache/compilation state remains
    shared; only model metadata belongs to the child.
    """
    from vllm import ModelRegistry

    child = copy.copy(parent)
    child.model_config = copy.deepcopy(parent.model_config)
    config = child.model_config
    config.model = checkpoint
    config.hf_config = hf_config
    config.hf_text_config = hf_config.get_text_config()
    config.hf_overrides = {}
    for name in ("is_encoder_decoder", "is_diffusion", "is_mm_prefix_lm"):
        config.__dict__.pop(name, None)
    config.model_arch_config = config.get_model_arch_config()
    config._model_info, _ = ModelRegistry.inspect_model_cls(config.architectures, config)
    return child
