# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Instance-local inference precision; BF16 bulk weights and exact FP32 router upcasts."""

from types import MethodType

import torch
import torch.nn.functional as F


def _head_forward(self, hidden):
    inputs = hidden.to(self.weight.dtype).reshape(-1, hidden.shape[-1]).contiguous()
    if inputs.is_cuda and inputs.dtype in (torch.float16, torch.bfloat16):
        logits = torch.mm(inputs, self.weight.t(), out_dtype=torch.float32)
        if self.bias is not None:
            logits.add_(self.bias.float())
    else:
        logits = F.linear(inputs.float(), self.weight.float(), self.bias.float() if self.bias is not None else None)
    return logits.reshape(*hidden.shape[:-1], self.weight.shape[0])


def _full_linear_forward(self, hidden):
    if hidden.is_cuda:
        from .grouped_mm import linear_fp32

        output = linear_fp32(hidden, self.weight)
        if self.bias is not None:
            output.add_(self.bias.float())
        return output.reshape(*hidden.shape[:-1], self.weight.shape[0])
    return F.linear(hidden.float(), self.weight.float(), self.bias.float() if self.bias is not None else None)


def _rms_forward(self, hidden):
    from .grouped_mm import rms_fp32

    return rms_fp32(hidden, self.weight, self.variance_epsilon)


def _router_forward(self, hidden):
    from .grouped_mm import linear_fp32

    hidden = hidden.reshape(-1, self.hidden_dim)
    logits = linear_fp32(hidden, self.weight)
    probabilities = logits.softmax(-1)
    values, indices = probabilities.topk(self.top_k, dim=-1)
    if self.norm_topk_prob:
        denominator = values[:, :1]
        for index in range(1, self.top_k):
            denominator = denominator + values[:, index : index + 1]
        values = values / denominator
    return logits, values, indices


def _float_embedding(module, args, output):
    return output.float()


def _rotary_output(module, args, output):
    return tuple(value.to(module._fusion_rotary_dtype) for value in output)


def _projection_output(module, args, output):
    return output.to(module._fusion_projection_dtype)


def _linear_input(module, args):
    weight = module.weight if hasattr(module, "weight") else module.gate_up_proj
    return (args[0].to(weight.dtype), *args[1:])


def _experts_forward(self, hidden, indices, weights):
    """Qwen3 MoE inference: BF16 GEMM operands, FP32 expert outputs/reduction."""
    from .grouped_mm import grouped_fp32

    count, width = hidden.shape
    top_k = indices.shape[-1]
    expert_ids, order = indices.reshape(-1).sort()
    counts = torch.histc(expert_ids.int(), bins=self.num_experts, min=0, max=self.num_experts - 1)
    offsets = counts.cumsum(0, dtype=torch.int32)
    dtype = torch.float32 if self._fusion_full_intermediate else self.gate_up_proj.dtype
    inputs = hidden[order // top_k].to(dtype)
    gate, up = grouped_fp32(inputs, self.gate_up_proj, offsets).chunk(2, dim=-1)
    activated = (self.act_fn(gate) * up).to(dtype)
    outputs = grouped_fp32(activated, self.down_proj, offsets)
    outputs.mul_(weights.reshape(-1)[order, None].float())
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    return outputs[inverse].view(count, top_k, width).sum(1)


def configure_precision(model, mode, *, attention_dtype=None):
    invariant_attention = mode in {"stable", "stable_moe_attention"}
    fixed_moe = mode in {"stable", "stable_moe", "stable_moe_attention"}
    if fixed_moe:
        mode = "stable"
    if mode == "stable":
        mode = "intermediate"
        attention_dtype = torch.float16
    if mode == "bf16":
        return
    if mode not in {"head", "residual", "linear", "intermediate"}:
        raise ValueError(f"Unknown inference precision: {mode}")
    if getattr(model, "_fusion_precision", None) is not None:
        raise ValueError("Inference precision has already been configured")
    model.lm_head.forward = MethodType(_head_forward, model.lm_head)
    if mode in {"residual", "linear", "intermediate"}:
        model.get_input_embeddings().register_forward_hook(_float_embedding)
        rotary = getattr(model.model, "rotary_emb", None)
        if rotary is not None:
            rotary._fusion_rotary_dtype = attention_dtype or model.get_input_embeddings().weight.dtype
            rotary.register_forward_hook(_rotary_output)
        for module in model.modules():
            if mode == "intermediate":
                continue
            weight = getattr(module, "weight", None)
            if isinstance(module, torch.nn.Linear) or (isinstance(weight, torch.Tensor) and weight.ndim == 2):
                if not isinstance(module, torch.nn.Embedding):
                    module.register_forward_pre_hook(_linear_input)
            elif hasattr(module, "gate_up_proj"):
                module.register_forward_pre_hook(_linear_input)
    if mode in {"linear", "intermediate"}:
        for module in model.modules():
            if type(module).__name__ == "Qwen3MoeExperts":
                module._fusion_full_intermediate = mode == "intermediate"
                module.forward = MethodType(_experts_forward, module)
            if hasattr(module, "top_k") and hasattr(module, "num_experts") and hasattr(module, "weight"):
                # Upcasting BF16 router parameters is exact, and keeps top-k
                # selection from changing due to a rounded BF16 router logit.
                module.weight.data = module.weight.data.float()
            if isinstance(module, torch.nn.Linear):
                module.forward = MethodType(_full_linear_forward if mode == "intermediate" else _head_forward, module)
        for layer in model.model.layers:
            attention = layer.self_attn
            for module in (attention.q_norm, attention.k_norm, attention.v_proj):
                module._fusion_projection_dtype = attention_dtype or attention.q_proj.weight.dtype
                module.register_forward_hook(_projection_output)
    model._fusion_precision = mode
    if fixed_moe and model.config.model_type == "qwen3_moe":
        for module in model.modules():
            if type(module).__name__ == "Qwen3MoeRMSNorm":
                module.forward = MethodType(_rms_forward, module)
            if type(module).__name__ == "Qwen3MoeTopKRouter":
                module.forward = MethodType(_router_forward, module)
        model._fusion_precision = "intermediate_fixed_moe"
        if invariant_attention:
            from .invariant_attention import NAME

            model._fusion_invariant_attention = True
            model.config._attn_implementation = NAME
            model._fusion_precision = "intermediate_invariant_moe"
