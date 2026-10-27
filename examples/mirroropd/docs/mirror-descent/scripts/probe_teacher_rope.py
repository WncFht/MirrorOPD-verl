#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Isolate whole-model dtype casting of real teacher RoPE buffers on GPU 0.

Weights and inputs remain fixed. This compares execution fidelity to the loaded
HF model, not task accuracy. No optimizer update or synthetic logits are used.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from verl.trainer.distillation.full_vocab import project
from verl.utils.torch_dtypes import cast_module_parameters


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--prefixes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.12)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.teacher, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        .cuda()
        .eval()
    )
    rope = model.model.rotary_emb
    original = rope.inv_freq.detach().clone()
    assert original.dtype == torch.float32
    parameters = {name: p.data_ptr() for name, p in model.named_parameters()}
    buffer_dtypes = {name: str(buffer.dtype) for name, buffer in model.named_buffers()}
    records = json.loads(args.prefixes.read_text())["records"][:4]
    baseline, rounded = [], []
    rows = []
    for mode in ("canonical", "module_bf16", "upcast_rounded", "restore_canonical", "parameter_only_cast"):
        if mode == "module_bf16":
            model.to(torch.bfloat16)
        elif mode == "upcast_rounded":
            rope.inv_freq = rope.inv_freq.float()
        elif mode == "restore_canonical":
            rope.inv_freq = original.clone()
            rope.original_inv_freq = original.clone()
        elif mode == "parameter_only_cast":
            cast_module_parameters(model, torch.bfloat16)
        assert all(
            p.dtype == torch.bfloat16 and p.data_ptr() == parameters[name] for name, p in model.named_parameters()
        )
        for index, record in enumerate(records):
            end = record["positions"][0] + 1 + 128
            tokens = torch.tensor(record["ids"][:end], device="cuda")[None]
            positions = torch.arange(record["positions"][0], record["positions"][0] + 128, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = (
                    model.model(input_ids=tokens, position_ids=torch.arange(end, device="cuda")[None], use_cache=False)
                    .last_hidden_state[0]
                    .index_select(0, positions)
                )
            lp = project(hidden, model.lm_head.weight).double().log_softmax(-1)
            if mode == "canonical":
                baseline.append(lp.cpu())
            reference = baseline[index].cuda()
            row = {
                "mode": mode,
                "prefix": index,
                "frequency_dtype": str(rope.inv_freq.dtype),
                "frequency_max_abs_change": (rope.inv_freq.float() - original).abs().max().item(),
                "kl_to_canonical": (lp.exp() * (lp - reference)).sum(-1).mean().item(),
                "kl_from_canonical": (reference.exp() * (reference - lp)).sum(-1).mean().item(),
                "logprob_max_abs_change": (lp - reference).abs().max().item(),
                "exact_canonical": torch.equal(lp, reference),
            }
            if mode == "module_bf16":
                rounded.append(lp.cpu())
            if mode == "upcast_rounded":
                row["exact_rounded"] = torch.equal(lp.cpu(), rounded[index])
                assert row["exact_rounded"]
            if mode in ("restore_canonical", "parameter_only_cast"):
                assert row["exact_canonical"]
            rows.append(row)
            print(mode, index, row["kl_to_canonical"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"scope": __doc__, "initial_buffer_dtypes": buffer_dtypes, "rows": rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
