# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Two native models, independent attention layers, and one FP32 logit mixture."""

import itertools
import math

import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.model_executor.models.utils import init_vllm_registered_model

from verl.experimental.rollout_policy.sampling import validate_tokenizers

from .compat import component_config, validate_component_families


class MirrorOPDFusionForCausalLM(nn.Module, MixtureOfExperts):
    """Text model with one TP group and one request driving both components."""

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        outer = vllm_config.model_config.hf_config
        if vllm_config.quant_config is not None or vllm_config.speculative_config is not None:
            raise ValueError("Native fusion has not accepted quantization or speculative decoding")
        if vllm_config.model_config.dtype != torch.bfloat16 or vllm_config.cache_config.cache_dtype not in {
            "auto",
            "bfloat16",
        }:
            raise ValueError("Native fusion requires BF16 weights and KV")
        if vllm_config.parallel_config.enable_eplb:
            raise ValueError("Native fusion has not accepted expert load balancing")
        if vllm_config.parallel_config.tensor_parallel_size not in {1, 2}:
            raise ValueError("Native fusion currently supports TP1/TP2")
        self.config = outer
        self.student_path = vllm_config.model_config.model
        self.teacher_path = outer.fusion_teacher_path
        student_config = AutoConfig.from_pretrained(self.student_path, local_files_only=True)
        teacher_config = AutoConfig.from_pretrained(self.teacher_path, local_files_only=True)
        validate_component_families(student_config, teacher_config)
        for config in (student_config, teacher_config):
            if vllm_config.model_config.max_model_len > config.max_position_embeddings:
                raise ValueError("Declared fusion context exceeds one component's native context")
            if getattr(config, "quantization_config", None):
                raise ValueError("Fusion checkpoints must be unquantized")
        student_tokenizer = AutoTokenizer.from_pretrained(self.student_path, local_files_only=True)
        teacher_tokenizer = AutoTokenizer.from_pretrained(self.teacher_path, local_files_only=True)
        legal = validate_tokenizers(
            student_tokenizer, teacher_tokenizer, student_config.vocab_size, teacher_config.vocab_size
        )
        mask = torch.zeros(student_config.vocab_size, dtype=torch.bool)
        mask[list(legal)] = True
        self.register_buffer("legal_tokens", mask, persistent=False)
        self.student_width = student_config.hidden_size
        self.student = init_vllm_registered_model(
            component_config(vllm_config, student_config, self.student_path), prefix=f"{prefix}.student".lstrip(".")
        )
        self.teacher = init_vllm_registered_model(
            component_config(vllm_config, teacher_config, self.teacher_path), prefix=f"{prefix}.teacher".lstrip(".")
        )
        self.num_moe_layers = getattr(self.teacher, "num_moe_layers", 0)
        self.expert_weights = []
        self.moe_layers = getattr(self.teacher, "moe_layers", [])
        if self.num_moe_layers:
            for name in (
                "num_expert_groups",
                "num_logical_experts",
                "num_physical_experts",
                "num_local_physical_experts",
                "num_routed_experts",
                "num_shared_experts",
                "num_redundant_experts",
            ):
                setattr(self, name, getattr(self.teacher, name))
        self.secondary_weights = [DefaultModelLoader.Source(self.teacher_path, None, prefix="teacher.")]
        self.set_alpha(getattr(outer, "fusion_teacher_weight", 0.5))

    def set_alpha(self, alpha):
        """Called only after admission closes and all requests have drained."""
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("Fusion alpha must be finite and in [0, 1]")
        self.alpha = float(alpha)

    def update_physical_experts_metadata(self, num_physical_experts, num_local_physical_experts):
        """Forward native MoE metadata; expert-parallel execution is not enabled."""
        self.teacher.update_physical_experts_metadata(num_physical_experts, num_local_physical_experts)
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts

    def embed_input_ids(self, input_ids):
        """Keep both native embeddings independent, including runner profiling."""
        return torch.cat((self.student.embed_input_ids(input_ids), self.teacher.embed_input_ids(input_ids)), dim=-1)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if intermediate_tensors is not None:
            raise ValueError("Fusion does not support pipeline intermediates")
        student_embeds = inputs_embeds[..., : self.student_width] if inputs_embeds is not None else None
        teacher_embeds = inputs_embeds[..., self.student_width :] if inputs_embeds is not None else None
        student = self.student(input_ids=input_ids, positions=positions, inputs_embeds=student_embeds)
        teacher = self.teacher(input_ids=input_ids, positions=positions, inputs_embeds=teacher_embeds)
        return torch.cat((student, teacher), dim=-1)

    def compute_logits(self, hidden_states):
        student = self.student.compute_logits(hidden_states[:, : self.student_width].contiguous())
        teacher = self.teacher.compute_logits(hidden_states[:, self.student_width :].contiguous())
        # Every TP rank enters BOTH head collectives before a non-output rank returns.
        if student is None or teacher is None:
            if student is not None or teacher is not None:
                raise RuntimeError("Fusion component head collectives disagree on their output rank")
            return None
        # Only align the sampling view. Both native heads/configs remain intact,
        # including teacher-only padding rows used by the independent RKL scorer.
        width = self.legal_tokens.numel()
        if student.shape[-1] != width:
            raise RuntimeError("Native student head disagrees with the declared sampling output width")
        teacher_view = teacher
        if teacher.shape[-1] > width:
            teacher_view = teacher[..., :width]
        elif teacher.shape[-1] < width:
            teacher_view = torch.nn.functional.pad(teacher, (0, width - teacher.shape[-1]), value=-torch.inf)
        if self.alpha == 0:
            logits = student.float()
        elif self.alpha == 1:
            logits = teacher_view.float()
        else:
            logits = student.float().mul(1 - self.alpha).add_(teacher_view.float(), alpha=self.alpha)
        if getattr(self, "capture_components", False) and self.alpha in {0, 1}:
            logits = logits.clone()
        logits.masked_fill_(~self.legal_tokens, -torch.inf)
        if getattr(self, "capture_components", False):
            self.component_logits = {
                "student": student.detach().float().clone(),
                "teacher": teacher.detach().float().clone(),
                "fusion": logits.detach().clone(),
                "legal": self.legal_tokens.detach().clone(),
            }
        return logits

    def load_weights(self, weights):
        """Use native loaders for primary student and secondary teacher checkpoints."""
        loaded = set()
        for is_teacher, group in itertools.groupby(weights, key=lambda row: row[0].startswith("teacher.")):
            component = "teacher" if is_teacher else "student"
            model = getattr(self, component)
            values = ((name.removeprefix("teacher.") if is_teacher else name, value) for name, value in group)
            loaded.update(f"{component}.{name}" for name in model.load_weights(values))
        return loaded
