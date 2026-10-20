#!/usr/bin/env python3
"""Reproduce verl's Qwen2 FLOP estimate from config and actual sequence lengths.

--uniform-length is a hypothetical equal-length batch; --lengths must be a JSON
list of global prompt+response lengths, never prompt-only/response-only lengths.
No Torch import or model weight reads are needed.
"""

import argparse
import json
from pathlib import Path


def architecture(config):
    if config["model_type"] != "qwen2":
        raise ValueError("This audit implements only the current Qwen2 estimator")
    h, v, layers = (config[k] for k in ("hidden_size", "vocab_size", "num_hidden_layers"))
    heads, kv, intermediate = (config[k] for k in ("num_attention_heads", "num_key_value_heads", "intermediate_size"))
    d = config.get("head_dim", h // heads)
    mlp = 3 * h * intermediate
    attn = h * (2 * heads + 2 * kv) * d
    embedding_head = 2 * v * h
    dense = (mlp + attn) * layers + embedding_head
    return {
        "mlp_per_layer": mlp,
        "attention_linear_per_layer": attn,
        "embedding_and_head": embedding_head,
        "dense_parameters": dense,
        "dense_flops_per_token": 6 * dense,
        "attention_flops_per_squared_length": 6 * layers * heads * d,
        "note": "Matches verl estimator, including its embedding and causal-attention approximation.",
    }


def flops(config, lengths):
    if not lengths or any(not isinstance(s, int) or s <= 0 for s in lengths):
        raise ValueError("lengths must be nonempty positive integers")
    a = architecture(config)
    t, q = sum(lengths), sum(s * s for s in lengths)
    dense = a["dense_flops_per_token"] * t
    attn = a["attention_flops_per_squared_length"] * q
    return {
        **a,
        "batch": len(lengths),
        "tokens": t,
        "squared_lengths": q,
        "dense_flops": dense,
        "attention_flops": attn,
        "total_flops": dense + attn,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--lengths", type=Path)
    group.add_argument("--uniform-length", type=int)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--actor-gpus", type=int, default=4)
    p.add_argument("--peak-tflops", type=float, required=True, help="one GPU's peak from verl device table")
    p.add_argument("--seconds", type=float, required=True, help="internal train_batch time, not outer update_actor")
    args = p.parse_args()
    if args.seconds <= 0 or args.actor_gpus <= 0 or args.peak_tflops <= 0 or args.batch <= 0:
        p.error("time, device count, peak and batch must be positive")
    config = json.loads(args.config.read_text())
    lengths = json.loads(args.lengths.read_text()) if args.lengths else [args.uniform_length] * args.batch
    result = flops(config, lengths)
    result.update(
        {
            "scenario": "actual-lengths" if args.lengths else "hypothetical-uniform-lengths",
            "seconds": args.seconds,
            "actor_gpus": args.actor_gpus,
            "peak_tflops": args.peak_tflops,
            "mfu": result["total_flops"] / (args.seconds * args.actor_gpus * args.peak_tflops * 1e12),
        }
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
