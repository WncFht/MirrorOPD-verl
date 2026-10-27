#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate and archive long-budget precision and single-factor interventions."""

import argparse
import json
import math
from pathlib import Path


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    arms = []
    alignment = identity = 0.0
    for run in ("long-precision", "factors", "muon-lr-controls", "precision-followup"):
        assert json.loads((args.root / f"{run}-exit.json").read_text())["returncode"] == 0
        raw = json.loads((args.root / run / "result.json").read_text())
        finite(raw)
        for arm in raw["branches"]:
            assert arm["steps"] == arm["optimizer_steps"] == len(arm["post_updates"])
            assert arm["frozen_weights_unchanged"]
            assert all(u["optim/actor/lr"] == arm["lr"] for u in arm["updates"])
            assert all(u["optim/actor/update_applied"] == 1 for u in arm["updates"])
            assert all(
                u["optim/actor/loss_nonfinite"] == u["optim/actor/grad_norm_preclip_nonfinite"] == 0
                for u in arm["updates"]
            )
            curve = []
            for post in arm["post_updates"]:
                m = post["metrics"]
                assert abs(m["distill/target_log_z"] - arm["before"]["distill/target_log_z"]) < 1e-12
                check = 0.3 * m["distill/teacher_kl"] + 0.7 * m["distill/anchor_kl"] + m["distill/target_log_z"]
                identity = max(identity, abs(check - m["distill/target_kl"]))
                if post["update"] < arm["steps"]:
                    alignment = max(
                        alignment, abs(m["distill/target_kl"] - arm["updates"][post["update"]]["distill/target_kl"])
                    )
                curve.append(
                    [
                        post["update"],
                        m["distill/target_kl"],
                        m["distill/teacher_kl"],
                        m["distill/anchor_kl"],
                        m["distill/mdpo/objective"],
                        post["grad_norm"],
                        post["delta_norm"],
                        post["gradient_dot_delta"],
                        arm["updates"][post["update"] - 1]["optim/actor/grad_norm_preclip"],
                    ]
                )
            if arm["lr"] == 0:
                assert all(post["delta_norm"] == 0 for post in arm["post_updates"])
                assert arm["before"]["distill/target_kl"] == arm["after"]["distill/target_kl"]
            if arm.get("save_best") and arm["best"]["update"] > 0:
                assert arm["best_optimizer_restored_exact"]
                assert (
                    abs(arm["best_rechecked"]["distill/target_kl"] - arm["best"]["metrics"]["distill/target_kl"])
                    < 1e-10
                )
            tail = [p[1] for p in curve[-32:]]
            arms.append(
                {
                    "run": run,
                    "index": arm["index"],
                    "solver": arm["solver"],
                    "lr": arm["lr"],
                    "precision": arm["precision"],
                    "rounded_initial": arm["rounded"],
                    "steps": arm["steps"],
                    "before": arm["before"],
                    "after": arm["after"],
                    "best": arm["best"],
                    "best_rechecked": arm.get("best_rechecked"),
                    "last_window": {
                        "n": len(tail),
                        "mean": sum(tail) / len(tail),
                        "below_initial": sum(v < arm["before"]["distill/target_kl"] for v in tail),
                    },
                    "post_series": curve,
                }
            )
    assert json.loads((args.root / "fixed-factors-exit.json").read_text())["returncode"] == 0
    forward = json.loads((args.root / "fixed-factors/result.json").read_text())
    assert json.loads((args.root / "attention-analysis-refined-exit.json").read_text())["returncode"] == 0
    attention = json.loads((args.root / "attention-analysis-refined.json").read_text())
    assert len(attention["rows"]) == 112
    head_analysis = json.loads((args.root / "attention-first-layer-heads.json").read_text())
    assert len(head_analysis["rows"]) == 4 and {row["rank"] for row in head_analysis["rows"]} == set(range(4))
    assert all(row["layer"] == 0 and len(row["query_heads"]) == 12 for row in head_analysis["rows"])
    value_offsets = json.loads((args.root / "attention-value-offsets.json").read_text())
    assert len(value_offsets["rows"]) == 12
    finite(head_analysis)
    finite(value_offsets)
    assert json.loads((args.root / "fixed-followup-exit.json").read_text())["returncode"] == 0
    forward_followup = json.loads((args.root / "fixed-followup/result.json").read_text())
    finite(forward_followup)
    finite(forward)
    finite(attention)
    boundary_summary = []
    for path in sorted((args.root / "boundaries").glob("rank*-*.json")):
        boundary = json.loads(path.read_text())
        mode, records = boundary["mode"], boundary["records"]
        for phase in ("grad", "no_grad"):
            head = records[f"{phase}/lm_head"]["input"][0]["dtype"]
            final_norm = records[f"{phase}/model.norm"]["output"]["dtype"]
            block = records[f"{phase}/model.layers.0"]["output"]["dtype"]
            inner_norm = records[f"{phase}/model.layers.0.input_layernorm"]["output"]["dtype"]
            rope = records[f"{phase}/model.rotary_emb"]["output"][0]["dtype"]
            assert head == final_norm == "torch.bfloat16", (path, phase)
            assert block == ("torch.float32" if mode == "residual_stream_fp32" else "torch.bfloat16")
            assert inner_norm == ("torch.float32" if mode == "norm_inner_fp32" else "torch.bfloat16")
            assert rope == ("torch.float32" if mode == "rope_table_math_fp32" else "torch.bfloat16")
            for layer in range(28):
                prefix = f"{phase}/model.layers.{layer}"
                assert records[prefix]["output"]["dtype"] == block
                for name in ("input_layernorm", "post_attention_layernorm"):
                    expected_norm = inner_norm
                    assert records[f"{prefix}.{name}"]["output"]["dtype"] == expected_norm
            boundary_summary.append(
                {
                    "file": str(path.relative_to(args.root)),
                    "layers_verified": 28,
                    "phase": phase,
                    "head_input": head,
                    "block_output": block,
                    "inner_norm_output": inner_norm,
                    "rope_output": rope,
                }
            )
        assert boundary["bf16_reduced_precision_reduction"] == (mode != "gemm_no_reduced")
    assert len(boundary_summary) == 4 * 14 * 2, len(boundary_summary)
    assert len(list((args.root / "followup-boundaries").glob("rank*-*.json"))) == 16
    for path in sorted((args.root / "followup-boundaries").glob("rank*-*.json")):
        boundary = json.loads(path.read_text())
        mode, records = boundary["mode"], boundary["records"]
        for phase in ("grad", "no_grad"):
            head = records[f"{phase}/lm_head"]["input"][0]["dtype"]
            norm = records[f"{phase}/model.norm"]["output"]["dtype"]
            block = records[f"{phase}/model.layers.0"]["output"]["dtype"]
            rope = records[f"{phase}/model.rotary_emb"]["output"][0]["dtype"]
            applied_rope = records[f"{phase}/model.layers.0.self_attn"]["position_embeddings"][0]["dtype"]
            assert head == norm == ("torch.float32" if mode == "midstream_head_fp32" else "torch.bfloat16")
            assert block == ("torch.float32" if mode.startswith("midstream") else "torch.bfloat16")
            assert rope == "torch.float32"
            assert applied_rope == ("torch.bfloat16" if mode == "rope_table_math_fp32" else "torch.float32")
            for layer in range(28):
                prefix = f"{phase}/model.layers.{layer}"
                assert records[prefix]["output"]["dtype"] == block
                expected_inner = "torch.float32" if mode.startswith("midstream") else "torch.bfloat16"
                for name in ("input_layernorm", "post_attention_layernorm"):
                    assert records[f"{prefix}.{name}"]["output"]["dtype"] == expected_inner
                assert all(
                    value["dtype"] == applied_rope for value in records[f"{prefix}.self_attn"]["position_embeddings"]
                )
            boundary_summary.append(
                {
                    "file": str(path.relative_to(args.root)),
                    "layers_verified": 28,
                    "phase": phase,
                    "head_input": head,
                    "block_output": block,
                    "rope_output": rope,
                    "applied_rope": applied_rope,
                }
            )
    assert all(replay["forward_equal_capture"] for row in attention["rows"] for replay in row["native_replay"].values())
    assert all(row["native_replay"]["True"]["dq_repeat_max_error_energy"] == 0 for row in attention["rows"])
    global_attention = {}
    for name in ("dq", "dk", "dv"):
        totals = {
            key: sum(row[name][key] for row in attention["rows"])
            for key in ("reference_energy", "error_energy", "cast_only_error_energy")
        }
        totals["relative_error"] = (totals["error_energy"] / totals["reference_energy"]) ** 0.5
        totals["cast_only_relative_error"] = (totals["cast_only_error_energy"] / totals["reference_energy"]) ** 0.5
        if name != "dv":
            for key in ("predicted_error_energy", "error_inner_product", "delta_error_residual_energy"):
                totals[key] = sum(row[name][key] for row in attention["rows"])
            totals["delta_error_cosine"] = (
                totals["error_inner_product"] / (totals["error_energy"] * totals["predicted_error_energy"]) ** 0.5
            )
        global_attention[name] = totals
    result = {
        "scope": (
            "State A, kappa=0 eta=0.3. Solver arms round the common initial masters once. "
            "Fixed-motion controls reuse older snapshots with matched effective BF16 weights. "
            "Attention captures hold Q/K/V/upstream fixed. Diagnostic interventions only."
        ),
        "base_commit": "ba9f2aa60",
        "post_series_columns": [
            "update",
            "target_kl",
            "teacher_kl",
            "anchor_kl",
            "objective",
            "grad_norm",
            "delta_norm",
            "gradient_dot_delta",
            "grad_norm_preclip",
        ],
        "arms": arms,
        "forward_controls": forward,
        "forward_followup": forward_followup,
        "attention_capture": attention,
        "attention_global": global_attention,
        "attention_first_layer_heads": head_analysis,
        "attention_value_offsets": value_offsets,
        "boundary_checks": boundary_summary,
        "checks": {
            "optimizer_calls": sum(a["steps"] for a in arms),
            "nonzero_lr_updates": sum(a["steps"] for a in arms if a["lr"] > 0),
            "native_gold_alignment_max_abs": alignment,
            "kl_identity_max_abs": identity,
            "max_grad_norm_preclip": max(p[8] for a in arms for p in a["post_series"]),
            "clipped_updates": sum(p[8] > 1 for a in arms for p in a["post_series"]),
        },
    }
    finite(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["checks"], indent=2))


if __name__ == "__main__":
    main()
