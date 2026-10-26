#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Archive complete solver interventions, excluding LR-mislabeled pilot arms."""

import argparse
import json
import math
import statistics
from pathlib import Path

SOLVER_RUNS = [
    "bf16-grid",
    "corrected-head",
    "long",
    "precision",
    "second-state",
    "long-exact",
    "mixed-precision",
    "muon-repeat",
    "second-confirm",
    "adam-long-controls",
    "projection-groups",
    "second-mixed",
    "residual-control",
    "second-fp32-arithmetic",
]
FORWARD_RUNS = ["mixed-fixed-step", "projection-fixed-step", "residual-fixed-step"]


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for v in value.values():
            finite(v)
    elif isinstance(value, list):
        for v in value:
            finite(v)


def summarize(root):
    arms, excluded, protocols = [], [], {}
    columns = [
        "update",
        "target_kl",
        "teacher_kl",
        "anchor_kl",
        "objective",
        "gradient_norm",
        "step_norm",
        "gradient_dot_step",
    ]
    total_updates = 0
    max_alignment = 0.0
    max_identity = 0.0
    for run in SOLVER_RUNS:
        status = json.loads((root / (run + "-exit.json")).read_text())
        assert status["returncode"] == 0, (run, status)
        data = json.loads((root / run / "result.json").read_text())
        finite(data)
        protocols[run] = {"arguments": data["arguments"], "checkpoint": data.get("checkpoint")}
        for r in data["branches"]:
            assert r["optimizer_steps"] == r["steps"]
            actual = sorted({u["optim/actor/lr"] for u in r["updates"]})
            if actual != [r["lr"]]:
                excluded.append(
                    {
                        "run": run,
                        "index": r["index"],
                        "solver": r["solver"],
                        "requested_lr": r["lr"],
                        "actual_lrs": actual,
                        "reason": "Pilot scheduler reused checkpoint initial_lr; the intended LR was rerun.",
                    }
                )
                continue
            assert all(
                u["optim/actor/update_applied"] == 1
                and u["optim/actor/loss_nonfinite"] == 0
                and u["optim/actor/grad_norm_preclip_nonfinite"] == 0
                for u in r["updates"]
            )
            if r.get("trainable", "all") != "all":
                assert r["frozen_weights_unchanged"]
            before = r["before"]
            after = r["after"]
            curve = []
            for post in r["post_updates"]:
                m = post["metrics"]
                assert abs(m["distill/target_log_z"] - before["distill/target_log_z"]) < 1e-12
                identity = 0.3 * m["distill/teacher_kl"] + 0.7 * m["distill/anchor_kl"] + m["distill/target_log_z"]
                max_identity = max(max_identity, abs(identity - m["distill/target_kl"]))
                if post["update"] < r["steps"]:
                    err = abs(m["distill/target_kl"] - r["updates"][post["update"]]["distill/target_kl"])
                    max_alignment = max(max_alignment, err)
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
                    ]
                )
            best = min([(0, before["distill/target_kl"])] + [(p[0], p[1]) for p in curve], key=lambda x: x[1])
            if r.get("save_best") and best[0] > 0:
                assert r["best_optimizer_restored_exact"]
                assert abs(r["best_rechecked"]["distill/target_kl"] - best[1]) < 1e-10
            tail = [p[1] for p in curve if p[0] > r["steps"] - 32]
            tail_summary = None
            if r["steps"] >= 32 and len(tail) == 32:
                tail_summary = {
                    "mean": statistics.mean(tail),
                    "median": statistics.median(tail),
                    "below_initial": sum(v < before["distill/target_kl"] for v in tail),
                    "count": 32,
                }
            arms.append(
                {
                    "run": run,
                    "index": r["index"],
                    "solver": r["solver"],
                    "lr": r["lr"],
                    "precision": r.get("precision", "bfloat16"),
                    "rounded_initial": r.get("rounded", False),
                    "trainable": r.get("trainable", "all"),
                    "steps": r["steps"],
                    "eval_interval": r.get("eval_interval", 1),
                    "before": before,
                    "after": after,
                    "best_gold_update": best[0],
                    "best_gold_target_kl": best[1],
                    "best_post_target_kl": min(p[1] for p in curve),
                    "native_preupdate_min": min(u["distill/target_kl"] for u in r["updates"]),
                    "best_rechecked": r.get("best_rechecked"),
                    "best_optimizer_restored_exact": r.get("best_optimizer_restored_exact"),
                    "last32": tail_summary,
                    "partition": r.get("partition"),
                    "clipped_updates": sum(u["optim/actor/grad_norm_preclip"] > 1 for u in r["updates"]),
                    "post_series": curve,
                }
            )
            total_updates += r["steps"]
    assert len(excluded) == 7, excluded
    forward = {}
    for name in FORWARD_RUNS:
        assert json.loads((root / (name + "-exit.json")).read_text())["returncode"] == 0
        forward[name] = json.loads((root / name / "result.json").read_text())
    finite(forward)
    warm = json.loads((root / "second-state/warmup.json").read_text())
    warm_steps = sum(len(r["updates"]) for r in warm["records"])
    return {
        "scope": (
            "Two fixed-prefix states; full-vocabulary MDPO kappa=0 eta=0.3. "
            "All solver results are independent restores. "
            "Precision comparisons use their own frozen initial policy. No formal task-quality or throughput claim."
        ),
        "base_commit": "fd5f4a479",
        "torch": "2.11.0+cu128",
        "post_series_columns": columns,
        "protocols": protocols,
        "arms": arms,
        "excluded_pilot_arms": excluded,
        "forward_controls": forward,
        "second_state_warmup": warm,
        "checks": {
            "included_successful_updates": total_updates,
            "warmup_successful_updates": warm_steps,
            "aligned_native_gold_max_abs": max_alignment,
            "kl_identity_max_abs": max_identity,
            "muon_cpu_reference": json.loads((root / "muon-cpu.json").read_text()),
            "muon_gpu_reference": json.loads((root / "muon-gpu.json").read_text()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["checks"], indent=2))


if __name__ == "__main__":
    main()
