#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Summarize saved optimizer interventions without interpolating missing results."""

import argparse
import json
import math
from pathlib import Path


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite result")
    if isinstance(value, dict):
        for v in value.values():
            finite(v)
    elif isinstance(value, list):
        for v in value:
            finite(v)


def lookup(run, precision, candidate):
    return next(r["metrics"] for r in run["evaluations"] if r["precision"] == precision and r["candidate"] == candidate)


def summarize(root):
    names = ["primary", "scale-sweep", "first64-fixed-input", "last64"]
    runs = {name: json.loads((root / name / "result.json").read_text()) for name in names}
    finite(runs)
    checks = {
        "target_identity_max_abs": 0.0,
        "objective_identity_max_abs": 0.0,
        "target_log_z_max_range": 0.0,
        "matched_norm_max_relative_error": 0.0,
    }
    for name, run in runs.items():
        arguments = run["arguments"]
        input_length = arguments.get("response_length", arguments["response_start"] + arguments["response_count"])
        if input_length != 128:
            raise ValueError(f"Model input length was not controlled: {name}")
        if not run["native_replica_exact"] or run["native_replica_relative_error"] != 0:
            raise ValueError(f"Native optimizer replay not exact: {name}")
        if len(run["evaluations"]) != (50 if name == "scale-sweep" else 22):
            raise ValueError(f"Incomplete precision matrix: {name}")
        zs = []
        for row in run["evaluations"]:
            m = row["metrics"]
            expected = 0.3 * m["distill/teacher_kl"] + 0.7 * m["distill/anchor_kl"] + m["distill/target_log_z"]
            checks["target_identity_max_abs"] = max(
                checks["target_identity_max_abs"], abs(expected - m["distill/target_kl"])
            )
            objective = m["distill/teacher_kl"] + (7 / 3) * m["distill/anchor_kl"]
            checks["objective_identity_max_abs"] = max(
                checks["objective_identity_max_abs"], abs(objective - m["distill/mdpo/objective"])
            )
            zs.append(m["distill/target_log_z"])
            if row["candidate"].endswith("@0.0") or row["candidate"] == "decay_only":
                if m != lookup(run, row["precision"], "initial"):
                    raise ValueError(f"Zero-displacement evaluation changed: {name}/{row['candidate']}")
        checks["target_log_z_max_range"] = max(checks["target_log_z_max_range"], max(zs) - min(zs))
        for candidate, stats in run["actual_updates"].items():
            if candidate.endswith("_matched"):
                checks["matched_norm_max_relative_error"] = max(
                    checks["matched_norm_max_relative_error"], abs(stats["relative_to_adam_norm"] - 1)
                )
    primary = runs["primary"]
    baseline = lookup(primary, "bfloat16", "initial")
    candidates = ["adam", "zero", "reset_m_matched", "fresh_matched", "current_increment_matched", "sgd_matched"]
    rows = []
    for c in candidates:
        m = lookup(primary, "bfloat16", c)
        dt = m["distill/teacher_kl"] - baseline["distill/teacher_kl"]
        dq = m["distill/anchor_kl"] - baseline["distill/anchor_kl"]
        rows.append(
            {
                "candidate": c,
                "target_kl": m["distill/target_kl"],
                "target_delta": m["distill/target_kl"] - baseline["distill/target_kl"],
                "teacher_delta": dt,
                "anchor_delta": dq,
                "fixed_update_eta_threshold": dq / (dq - dt) if dt < 0 and dq > 0 else None,
            }
        )
    scan = runs["scale-sweep"]
    nonzero = [
        r
        for r in scan["evaluations"]
        if r["precision"] == "bfloat16" and r["candidate"] != "initial" and not r["candidate"].endswith("@0.0")
    ]
    scan_summary = {
        "bf16_nonzero_candidates": len(nonzero),
        "bf16_nonzero_improved": sum(
            r["metrics"]["distill/target_kl"] < baseline["distill/target_kl"] for r in nonzero
        ),
        "bf16_nonzero_target_range": [
            min(r["metrics"]["distill/target_kl"] for r in nonzero),
            max(r["metrics"]["distill/target_kl"] for r in nonzero),
        ],
    }
    checks["partition_consistency_max_abs"] = 0.0
    for c in ["initial", "zero", "decay_only"]:
        for precision in ["bfloat16", "float32"]:
            all_rows = lookup(primary, precision, c)
            first = lookup(runs["first64-fixed-input"], precision, c)
            last = lookup(runs["last64"], precision, c)
            for k in all_rows:
                checks["partition_consistency_max_abs"] = max(
                    checks["partition_consistency_max_abs"], abs((first[k] + last[k]) / 2 - all_rows[k])
                )
    assert checks["target_identity_max_abs"] < 1e-12
    assert checks["objective_identity_max_abs"] < 1e-12
    assert checks["target_log_z_max_range"] == 0
    assert checks["matched_norm_max_relative_error"] < 0.002
    assert checks["partition_consistency_max_abs"] < 1e-9
    return {
        "scope": (
            "Single native gradient per loss mask, identical model/Adam, independent one-step counterfactuals. "
            "Masks share full 128-token inputs. FP32 forward uses BF16-valued weights. Shared GPUs, no timing claim."
        ),
        "source_commit": "c4b5812bb",
        "runs": runs,
        "checks": checks,
        "primary_summary": rows,
        "scale_summary": scan_summary,
        "excluded_exploration": (
            "first64 used a truncated 64-token input and is superseded by first64-fixed-input "
            "to hold attention shapes fixed."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checks": result["checks"], "scan": result["scale_summary"]}, indent=2))


if __name__ == "__main__":
    main()
