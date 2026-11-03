# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Per-instance causal-LM output adapter; preserve root AND head FSDP hooks."""

from types import MethodType


def install_full_vocab_adapter(model):
    if hasattr(model, "_full_vocab_original_forward"):
        return
    linear_head_layout(model.config)
    if model.get_output_embeddings() is not model.lm_head or not hasattr(model, "model"):
        raise ValueError("Unsupported backbone/head layout")
    if model.lm_head.bias is not None:
        raise ValueError("full reverse KL requires a bias-free LM head")
    if model.config.model_type == "mimo":
        # Published MiMo inherits the Qwen2 next-token forward. Its MTP modules
        # are unused by this objective and remain frozen in every study arm.
        for parameter in model.model.mtp_layers.parameters():
            parameter.requires_grad_(False)
    model._full_vocab_original_forward = model.forward
    model.lm_head._full_vocab_original_forward = model.lm_head.forward
    model.forward = MethodType(full_vocab_forward, model)
    model.lm_head.forward = MethodType(full_vocab_head_forward, model.lm_head)


def full_vocab_head_forward(self, hidden, *, full_vocab_callback=None):
    if full_vocab_callback is None:
        return self._full_vocab_original_forward(hidden)
    if self.bias is not None:
        raise ValueError("full reverse KL currently requires a bias-free LM head")
    return full_vocab_callback(hidden, self.weight)


def full_vocab_forward(self, *args, full_vocab_callback=None, **kwargs):
    if full_vocab_callback is None:
        return self._full_vocab_original_forward(*args, **kwargs)
    kwargs.update(return_dict=True, output_hidden_states=False, use_cache=False)
    from verl.trainer.distillation.profiling import full_kl_range

    with full_kl_range("backbone"):
        output = self.model(*args, **kwargs)
    return full_vocab_callback(output.last_hidden_state, self.lm_head)


def linear_head_layout(config):
    """Validated HF families with an unscaled, bias-free linear output head."""
    get = config.get if isinstance(config, dict) else lambda key, default=None: getattr(config, key, default)
    if get("model_type") not in ("qwen2", "qwen3", "qwen3_moe", "llama", "mimo", "olmo3"):
        raise ValueError("Full-vocab adapter supports Qwen2/Qwen3/Qwen3-MoE/Llama/MiMo/Olmo3")
    if get("quantization_config") or get("final_logit_softcapping") or get("logit_scale", 1) not in (None, 1):
        raise ValueError("Quantized, softcapped or scaled output heads need their own projection adapter")
    if get("lm_head_bias", False):
        raise ValueError("Full-vocab adapter requires a bias-free linear output head")
    return "model.embed_tokens.weight" if get("tie_word_embeddings", False) else "lm_head.weight"
