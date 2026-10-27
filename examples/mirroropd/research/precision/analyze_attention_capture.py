#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare actual captured FA2 VJPs with a causal FP64 conditional reference.

No model update or synthetic input is used. A delta-only prediction recomputes
P in FP64 but uses the captured rounded output in sum(dO * O). Its agreement
with the actual gradient error is descriptive, not a causal attribution.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .factor_mirror_precision import dense_attention_vjp


def compare(actual, reference, prediction=None):
    error = actual.double() - reference
    energy = reference.square().sum().item()
    ee = error.square().sum().item()
    result = {
        "reference_energy": energy,
        "error_energy": ee,
        "relative_error": (ee / energy) ** 0.5 if energy else None,
    }
    cast_error = reference.to(actual.dtype).double() - reference
    result["cast_only_error_energy"] = cast_error.square().sum().item()
    if prediction is not None:
        predicted_error = prediction - reference
        pe = predicted_error.square().sum().item()
        inner = (predicted_error * error).sum().item()
        result.update(
            predicted_error_energy=pe,
            error_inner_product=inner,
            delta_error_cosine=inner / (pe * ee) ** 0.5 if pe and ee else None,
            delta_error_residual_energy=(error - predicted_error).square().sum().item(),
        )
    return result


def replay_native(record, q, k, v, go):
    """Repeat the installed HF/FA2 route with the captured packing arguments."""
    from transformers.integrations.flash_attention import flash_attention_forward

    module = SimpleNamespace(
        layer_idx=record["layer"], is_causal=True, config=SimpleNamespace(_attn_implementation="flash_attention_2")
    )
    kwargs = {
        key: value.to(q.device) if isinstance(value, torch.Tensor) else value
        for key, value in record["native_kwargs"].items()
    }
    captured = record["dq"].to(q.device).double()
    energy = captured.square().sum().item()
    result = {}
    for deterministic in (False, True):
        repeats = []
        forward_equal = True
        for _ in range(3):
            with torch.enable_grad(), torch.autocast(q.device.type, dtype=q.dtype):
                inputs = [x.detach().requires_grad_() for x in (q, k, v)]
                output, _ = flash_attention_forward(
                    module, *inputs, None, **(kwargs | {"deterministic": deterministic})
                )
                gradients = torch.autograd.grad(output, inputs, go)
            forward_equal &= torch.equal(output.detach(), record["out"].to(q.device))
            repeats.append(gradients[0].double())
        repeat_error = max((g - repeats[0]).square().sum().item() for g in repeats[1:])
        result[str(deterministic)] = {
            "forward_equal_capture": forward_equal,
            "dq_repeat_max_error_energy": repeat_error,
            "dq_repeat_max_relative": (repeat_error / energy) ** 0.5 if energy else None,
            "dq_difference_from_capture_energy": (repeats[0] - captured).square().sum().item(),
        }
    return result


@torch.no_grad()
def analyze(path, device):
    record = torch.load(path, map_location="cpu", weights_only=True)
    q, k, v, go, out = [record[n].to(device) for n in ("q", "k", "v", "go", "out")]
    reference = dense_attention_vjp(q, k, v, go, record["scale"])
    groups = q.shape[1] // k.shape[1]
    qh, kh, vh = q.double(), k.double().repeat_interleave(groups, 1), v.double().repeat_interleave(groups, 1)
    gh = go.transpose(1, 2).double()
    scores = qh @ kh.transpose(-1, -2) * record["scale"]
    scores.masked_fill_(torch.ones(scores.shape[-2:], device=device, dtype=torch.bool).triu(1), -torch.inf)
    p = scores.softmax(-1)
    reference_output = p @ vh
    dp = gh @ vh.transpose(-1, -2)
    true_delta = (p * dp).sum(-1, keepdim=True)
    rounded_delta = (gh * out.transpose(1, 2).double()).sum(-1, keepdim=True)
    rounded_delta_fp32 = (go.transpose(1, 2).float() * out.transpose(1, 2).float()).sum(-1, keepdim=True).double()
    shifted_ds = p * (dp - rounded_delta)
    predicted_dq = shifted_ds @ kh * record["scale"]
    ideal_cast_delta = (gh * reference_output.to(out.dtype).double()).sum(-1, keepdim=True)
    ideal_cast_dq = (p * (dp - ideal_cast_delta)) @ kh * record["scale"]
    cast_prediction = ideal_cast_dq - reference[0]
    generation_prediction = predicted_dq - ideal_cast_dq
    predicted_dk = shifted_ds.transpose(-1, -2) @ qh * record["scale"]
    predicted_dk = predicted_dk.reshape(k.shape[0], k.shape[1], groups, k.shape[2], k.shape[3]).sum(2)
    actual = [record[n].to(device) for n in ("dq", "dk", "dv")]
    dq_norm = reference[0].square().sum(-1).sqrt()
    error_norm = (actual[0].double() - reference[0]).square().sum(-1).sqrt()
    positive = dq_norm > 0
    row_relative = error_norm[positive] / dq_norm[positive]
    result = {
        "rank": record["rank"],
        "layer": record["layer"],
        "query_shape": list(q.shape),
        "dq": compare(actual[0], reference[0], predicted_dq),
        "dk": compare(actual[1], reference[1], predicted_dk),
        "dv": compare(actual[2], reference[2]),
        "output": compare(out.transpose(1, 2), reference_output),
        "delta_error_norm": (rounded_delta - true_delta).norm().item(),
        "delta_error_sum": (rounded_delta - true_delta).sum().item(),
        "delta_positive_error_rows": (rounded_delta > true_delta).sum().item(),
        "delta_negative_error_rows": (rounded_delta < true_delta).sum().item(),
        "delta_rows": true_delta.numel(),
        "delta_fp32_dot_error_norm": (rounded_delta_fp32 - rounded_delta).norm().item(),
        "dq_output_reference_split": {
            "ideal_output_cast_prediction_energy": cast_prediction.square().sum().item(),
            "native_minus_ideal_cast_prediction_energy": generation_prediction.square().sum().item(),
            "twice_cross_inner_product": (2 * cast_prediction * generation_prediction).sum().item(),
            "total_prediction_energy": (predicted_dq - reference[0]).square().sum().item(),
        },
        "dq_zero_reference_rows": (~positive).sum().item(),
        "dq_nonzero_in_zero_reference_rows": ((~positive) & (error_norm > 0)).sum().item(),
        "dq_zero_reference_error_energy": error_norm[~positive].square().sum().item(),
        "dq_row_relative_quantiles": torch.quantile(
            row_relative, row_relative.new_tensor([0.5, 0.9, 0.99, 1])
        ).tolist(),
        "dq_first_position_native_norm": actual[0][:, :, 0].double().norm().item(),
        "dq_first_position_reference_norm": reference[0][:, :, 0].norm().item(),
        "query_heads": [
            {
                "head": head,
                "reference_energy": reference[0][:, head].square().sum().item(),
                "error_energy": (actual[0][:, head].double() - reference[0][:, head]).square().sum().item(),
                "mean_max_probability": p[:, head].amax(-1).mean().item(),
                "mean_first_key_probability": p[:, head, :, 0].mean().item(),
                "mean_entropy": -torch.special.xlogy(p[:, head], p[:, head]).sum(-1).mean().item(),
            }
            for head in range(q.shape[1])
        ],
    }
    if q.is_cuda and "native_kwargs" in record:
        result["native_replay"] = replay_native(record, q, k, v, go)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", type=int, nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device == "cuda":
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(0.05)
    files = sorted(args.capture.glob("rank*-layer*.pt"))
    assert len(files) == 4 * 28, len(files)
    if args.layers is not None:
        assert len(set(args.layers)) == len(args.layers) and all(0 <= layer < 28 for layer in args.layers)
        files = [p for p in files if int(p.stem.split("-layer")[1]) in args.layers]
        assert len(files) == 4 * len(args.layers)
    rows = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for path in files:
        rows.append(analyze(path, args.device))
        args.out.write_text(json.dumps({"scope": __doc__, "rows": rows}, indent=2) + "\n")
        print(path.name, rows[-1]["dq"]["relative_error"], rows[-1]["dq"]["delta_error_cosine"], flush=True)


if __name__ == "__main__":
    main()
