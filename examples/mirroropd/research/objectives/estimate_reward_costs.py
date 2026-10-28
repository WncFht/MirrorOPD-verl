# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Reproduce tensor-size budgets and conditional timing scenarios; no GPU needed."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--response-length", type=int, default=7168)
    parser.add_argument("--prompt-length", type=int, default=4096)
    parser.add_argument("--micro-tokens", type=int, default=32768)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert min(args.batch, args.response_length, args.prompt_length, args.micro_tokens) > 0
    docs = Path(__file__).resolve().parents[2] / "docs"
    model = json.loads((docs / "data/student-config.json").read_text())
    measured = json.loads((docs / "data/local-shared4-20261021/analysis.json").read_text())["runs"]["graph"]
    v, h = model["vocab_size"], model["hidden_size"]
    gib, mib = 2**30, 2**20
    n = args.batch * args.response_length
    all_positions = args.batch * (args.response_length + args.prompt_length)
    mean_response = measured["mean_metrics"]["response_length/mean"]
    n_observed = args.batch * mean_response
    payloads = []
    for name, width_bytes in [
        ("sampled_logprob_only_fp32", 4),
        ("sampled_id32_logprob32", 8),
        *[(f"teacher_top{k}_id32_logprob32", k * 8) for k in [4, 16, 32, 128]],
        *[(f"teacher_top{k}_plus_sample_id32_logprob32", (k + 1) * 8) for k in [4, 16, 32, 128]],
        ("teacher_hidden_bf16", h * 2),
        ("teacher_logits_bf16", v * 2),
        ("teacher_logprobs_fp32", v * 4),
    ]:
        payloads.append(
            dict(
                name=name,
                bytes_per_position=width_bytes,
                response_cap_global_mib=n * width_bytes / mib,
                response_cap_per_gpu_gib={str(g): n * width_bytes / g / gib for g in [4, 8]},
                observed_response_global_mib=n_observed * width_bytes / mib,
                full_sequence_cap_global_mib=all_positions * width_bytes / mib,
            )
        )
    stages = measured["stage_seconds"]
    t0 = measured["mean_metrics"]["timing_s/step"]
    scenarios = []
    for epochs in [1, 4]:
        same_cost_baseline = t0 + (epochs - 1) * stages["update_actor"]
        for factor in [1.0, 1.1, 1.5, 2.0]:
            predicted = same_cost_baseline + (factor - 1) * (
                stages["teacher_forward"] + epochs * stages["update_actor"]
            )
            scenarios.append(
                dict(
                    ppo_epochs=epochs,
                    teacher_and_actor_factor=factor,
                    conditional_seconds=predicted,
                    baseline_seconds=same_cost_baseline,
                    increase_percent=(predicted / same_cost_baseline - 1) * 100,
                )
            )
    data = dict(
        assumptions=dict(
            batch=args.batch,
            response_cap=args.response_length,
            prompt_cap=args.prompt_length,
            vocab=v,
            hidden=h,
            micro_token_budget=args.micro_tokens,
            observed_mean_response_length=mean_response,
            observed_response_tokens=n_observed,
            observed_total_tokens=measured["mean_metrics"]["perf/total_num_tokens"],
            source="docs/archive/evidence/root-data/local-shared4-20261021/analysis.json; H800x4, Graph, steps 2-5",
            note="Payload sizes are raw tensors, not GPU peak memory or serialized traffic.",
        ),
        payloads=payloads,
        working_set=dict(
            micro_one_bf16_nv_gib=args.micro_tokens * v * 2 / gib,
            micro_one_fp32_nv_gib=args.micro_tokens * v * 4 / gib,
            naive_two_bf16_plus_three_fp32_nv_gib=args.micro_tokens * v * 16 / gib,
            teacher_head_bf16_gib=v * h * 2 / gib,
            teacher_head_fp32_gib=v * h * 4 / gib,
            teacher_hidden_micro_bf16_mib=args.micro_tokens * h * 2 / mib,
            chunk512_one_fp32_vocab_gib=512 * v * 4 / gib,
            tile512x4096_one_fp32_mib=512 * 4096 * 4 / mib,
            hidden_vs_logits_compression=v / h,
            teacher_head_forward_response_cap_tflops=2 * n * h * v / 1e12,
        ),
        timing=dict(
            reference_batch=64,
            reference_response_cap=7168,
            applies_to_requested_sizes=(
                args.batch == 64
                and args.response_length == 7168
                and args.prompt_length == 4096
                and args.micro_tokens == 32768
            ),
            measured_h800x4_seconds=t0,
            stages=stages,
            scenarios=scenarios,
            note="Factors are sensitivity assumptions, not measured method speedups; transfer/startup/eval excluded.",
        ),
        transfer_lower_bounds=[
            dict(
                payload=name,
                aggregate_gib_per_s=bw,
                one_leg_seconds=n * size / gib / bw,
                two_legs_seconds=2 * n * size / gib / bw,
            )
            for name, size in [("bf16_logits", v * 2), ("bf16_hidden", h * 2), ("top16_plus_sample", 17 * 8)]
            for bw in [25, 50, 100]
        ],
    )
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
