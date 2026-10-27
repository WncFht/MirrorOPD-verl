# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Instance-local execution policies for dense Qwen2/Qwen3 training.

The mixed profile keeps BF16 linear operands and attention, with FP32 linear
outputs, residuals, norms and head computation. Parameter names/objects and
FSDP hooks are unchanged. This is an execution policy, independent of the loss.
"""

from types import MethodType

import torch
import torch.nn.functional as F

from verl.utils.linear import project


def _embedding_forward(self, ids):
    return F.embedding(
        ids, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
    ).float()


def _linear_forward(self, inputs):
    with torch.autocast(inputs.device.type, enabled=False):
        operands = inputs.reshape(-1, inputs.shape[-1]).to(self.weight.dtype)
        result = project(operands, self.weight).float().reshape(*inputs.shape[:-1], self.out_features)
        return result if self.bias is None else result + self.bias.float()


def _head_forward(self, inputs):
    with torch.autocast(inputs.device.type, enabled=False):
        return F.linear(inputs.float(), self.weight.float(), None if self.bias is None else self.bias.float())


def packed_fp32_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    *,
    scaling=None,
    dropout=0.0,
    position_ids=None,
    sliding_window=None,
    **kwargs,
):
    """Reference attention: separate packed sequences, with FP32 SDPA math.

    No dense cross-sequence mask or implicit conversion to half precision.
    Intentionally slower than FlashAttention; intended as a correctness reference.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if query.dtype != torch.float32 or key.dtype != torch.float32 or value.dtype != torch.float32:
        raise ValueError("FP32 reference attention received low-precision Q/K/V")
    if attention_mask is not None or sliding_window is not None or query.shape[2] != key.shape[2]:
        raise ValueError("FP32 packed reference requires causal self-attention without padding/sliding window/cache")
    if position_ids is None:
        raise ValueError("FP32 packed reference requires position_ids")
    if query.shape[0] != 1 or position_ids.shape != (1, query.shape[2]):
        raise ValueError("FP32 packed reference expects one packed row with matching position_ids")
    boundaries = (torch.where(position_ids[0, 1:] <= position_ids[0, :-1])[0] + 1).tolist()
    offsets = [0, *boundaries, query.shape[2]]
    pieces = []
    with sdpa_kernel(SDPBackend.MATH):
        for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
            pieces.append(
                F.scaled_dot_product_attention(
                    query[:, :, start:stop],
                    key[:, :, start:stop],
                    value[:, :, start:stop],
                    dropout_p=dropout,
                    is_causal=True,
                    scale=scaling,
                    enable_gqa=True,
                )
            )
    return torch.cat(pieces, dim=2).transpose(1, 2).contiguous(), None


def register_fp32_attention():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface

    AttentionInterface.register("verl_fp32", packed_fp32_attention)
    # Sequence boundaries and causality are handled inside the reference kernel.
    AttentionMaskInterface.register("verl_fp32", lambda **kwargs: None)


def prepare_precision_config(model_config, engine_config):
    """Validate before loading weights; do not patch any HF model class."""
    mode = engine_config.forward_precision
    if mode is None:
        return
    if (
        engine_config.strategy != "fsdp2"
        or engine_config.ulysses_sequence_parallel_size != 1
        or engine_config.qat.enable
    ):
        raise ValueError("forward_precision profiles require FSDP2, SP=1, and no QAT")
    from verl.utils.torch_dtypes import PrecisionType

    if not engine_config.forward_only and engine_config.model_dtype is not None:
        if PrecisionType.to_dtype(engine_config.model_dtype) != torch.float32:
            raise ValueError("Training precision profiles require FP32 master parameters")
    if mode in ("fp32", "bf16_fp32"):
        torch.set_float32_matmul_precision("highest")
    if model_config.hf_config.model_type not in ("qwen2", "qwen3"):
        raise ValueError("forward_precision profiles currently support dense Qwen2/Qwen3")
    if model_config.use_liger or model_config.use_fused_kernels or model_config.lora_rank > 0:
        raise ValueError("forward_precision profiles require native modules, no Liger/fused PPO/LoRA")
    if mode == "bf16_fp32" and model_config.hf_config._attn_implementation != "flash_attention_2":
        raise ValueError("bf16_fp32 requires FlashAttention2 for its low-precision attention boundary")
    if mode == "fp32":
        if not model_config.use_remove_padding:
            raise ValueError("FP32 reference requires packed input")
        if getattr(model_config.hf_config, "sliding_window", None) is not None:
            raise ValueError("FP32 reference does not support sliding-window attention")
        register_fp32_attention()
        model_config.hf_config._attn_implementation = "verl_fp32"


def install_precision_adapter(model, mode):
    if mode != "bf16_fp32" or getattr(model, "_verl_precision_adapter", None) == mode:
        return
    # Instance methods preserve parameter identities, state-dict names, tied
    # embeddings and the root/head FSDP call boundaries.
    for name, module in model.named_modules():
        if type(module) is torch.nn.Embedding:
            module.forward = MethodType(_embedding_forward, module)
        elif type(module) is torch.nn.Linear:
            module.forward = MethodType(_head_forward if name == "lm_head" else _linear_forward, module)
    model._verl_precision_adapter = mode
