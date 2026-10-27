#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Real-tensor value-offset counterfactual, without any model optimization.

For row-stochastic P, P(V-c)+c = PV, and the Q/K VJP is invariant to a
key-independent c. Compare native FA2 with first-value and rounded-mean shifts, holding
Q/K/dO fixed. Whole-sequence means are diagnostic only, not an inference policy.
Report centering quantization separately. This is a local
diagnostic, not a claim that a centered production kernel fixes MDPO.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from factor_mirror_precision import dense_attention_vjp
from transformers.integrations.flash_attention import flash_attention_forward


def energy(x):
    return x.double().square().sum().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.05)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)
    rows = []
    for rank in range(4):
        record = torch.load(args.capture / f"rank{rank}-layer0.pt", map_location="cpu", weights_only=True)
        q, k, v, go = [record[name].cuda() for name in ("q", "k", "v", "go")]
        reference = dense_attention_vjp(q, k, v, go, record["scale"])
        groups = q.shape[1] // k.shape[1]
        output_reference = torch.nn.functional.scaled_dot_product_attention(
            q.double(),
            k.double().repeat_interleave(groups, 1),
            v.double().repeat_interleave(groups, 1),
            is_causal=True,
            scale=record["scale"],
        ).transpose(1, 2)
        module = SimpleNamespace(
            layer_idx=0, is_causal=True, config=SimpleNamespace(_attn_implementation="flash_attention_2")
        )
        kwargs = {
            key: value.cuda() if isinstance(value, torch.Tensor) else value
            for key, value in record["native_kwargs"].items()
        }
        for mode in ("native", "subtract_first_value", "subtract_rounded_mean_value"):
            if mode == "native":
                offset = torch.zeros_like(v[:, :, :1]).float()
            elif mode == "subtract_first_value":
                offset = v[:, :, :1].float()
            else:
                offset = v.float().mean(-2, keepdim=True).bfloat16().float()
            exact_centered = v.double() - offset.double()
            centered = exact_centered.to(v.dtype)
            conditional = dense_attention_vjp(q, k, centered, go, record["scale"])
            inputs = [x.detach().requires_grad_() for x in (q, k, centered)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output, _ = flash_attention_forward(module, *inputs, None, **(kwargs | {"deterministic": True}))
            gradients = torch.autograd.grad(output, inputs, go)
            restored = output.detach().float() + offset.repeat_interleave(groups, 1).transpose(1, 2)
            if mode == "native":
                assert torch.equal(output.detach(), record["out"].cuda())
            rows.append(
                {
                    "rank": rank,
                    "layer": 0,
                    "mode": mode,
                    "value_energy": energy(v),
                    "centered_value_energy": energy(exact_centered),
                    "centering_quantization_error_energy": energy(centered.double() - exact_centered),
                    "output_reference_energy": energy(output_reference),
                    "restored_output_fp32_error_energy": energy(restored.double() - output_reference),
                    "restored_output_bf16_error_energy": energy(restored.bfloat16().double() - output_reference),
                    "gradients": {
                        name: {
                            "reference_energy": energy(ref),
                            "error_energy": energy(actual.double() - ref),
                            "own_centered_reference_error_energy": energy(actual.double() - own),
                            "centering_reference_change_energy": energy(own - ref),
                        }
                        for name, actual, ref, own in zip(
                            ("dq", "dk", "dv"), gradients, reference, conditional, strict=True
                        )
                    },
                }
            )
            print(rank, mode, (rows[-1]["gradients"]["dq"]["error_energy"] / energy(reference[0])) ** 0.5, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"scope": __doc__, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
