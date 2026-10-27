#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of installed Qwen2 dtype semantics, not a GPU training experiment.

The small tensors validate dtype propagation and rounding examples. They do not
measure the contribution of an operator to the real checkpoint's KL rebound.
"""

import argparse
import inspect
import json
from pathlib import Path

import torch
import transformers
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2RotaryEmbedding, apply_rotary_pos_emb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    config = Qwen2Config(hidden_size=64, num_attention_heads=4, num_key_value_heads=4, intermediate_size=128)
    rope = Qwen2RotaryEmbedding(config)
    q = torch.randn(1, 4, 8, 16).bfloat16()
    positions = torch.arange(8)[None]
    propagation = []
    for dtype in (torch.bfloat16, torch.float32):
        x = torch.randn(1, 8, 64).to(dtype)
        cos, sin = rope(x, positions)
        rotated, _ = apply_rotary_pos_emb(q, q, cos, sin)
        norm = Qwen2RMSNorm(64).bfloat16()
        propagation.append(
            {
                "embedding_output": str(dtype),
                "frequency_buffer": str(rope.inv_freq.dtype),
                "cos_sin_output": str(cos.dtype),
                "query_input": str(q.dtype),
                "rotated_query": str(rotated.dtype),
                "rmsnorm_output": str(norm(x).dtype),
            }
        )
    assert propagation[0]["rotated_query"] == "torch.bfloat16"
    assert propagation[1]["rotated_query"] == "torch.float32"
    original = rope.inv_freq.clone()
    rope.bfloat16().float()
    buffer_error = (rope.inv_freq - original).abs().max().item()
    assert buffer_error > 0
    cases = []
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        one = torch.tensor(1.0, dtype=dtype)
        delta = torch.tensor(0.001, dtype=dtype)
        actual = ((one + delta) - one).item()
        cases.append({"dtype": str(dtype), "eps": torch.finfo(dtype).eps, "realized_residual_increment": actual})
    bf16_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    result = {
        "scope": __doc__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "fresh_process_bf16_reduced_precision_reduction": bf16_reduction,
        "qwen2_dtype_propagation": propagation,
        "rope_buffer_bf16_then_fp32_max_error": buffer_error,
        "illustrative_addition_one_plus_0_001": cases,
        "source": {
            "rope_forward": inspect.getsource(Qwen2RotaryEmbedding.forward),
            "rmsnorm_forward": inspect.getsource(Qwen2RMSNorm.forward),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source"}, indent=2))


if __name__ == "__main__":
    main()
