#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Archive ordinary OPD comparisons and the teacher-buffer repair controls."""

import argparse
import json
from pathlib import Path

from examples.mirroropd.research.core.io import finite

RUNS = (
    "state-a",
    "state-b",
    "adam-confirm",
    "opd-continuation",
    "teacher-rounded",
    "teacher-canonical",
    "teacher-native-fixed",
    "mdpo-canonical",
    "teacher-canonical-fp32",
    "mdpo-canonical-fp32",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    arms, protocols = [], {}
    alignment = identity = 0.0
    for run in RUNS:
        assert json.loads((args.root / f"{run}-exit.json").read_text())["returncode"] == 0
        raw = json.loads((args.root / run / "result.json").read_text())
        finite(raw)
        protocols[run] = {
            "arguments": raw["arguments"],
            "teacher_rope": raw["teacher_rope"],
            "teacher_equals_canonical": raw.get("teacher_equals_canonical"),
        }
        for arm in raw["branches"]:
            assert arm["optimizer_steps"] == arm["steps"] == len(arm["post_updates"])
            assert all(
                u["optim/actor/lr"] == arm["lr"] and u["optim/actor/update_applied"] == 1 for u in arm["updates"]
            )
            assert all(
                u["optim/actor/loss_nonfinite"] == u["optim/actor/grad_norm_preclip_nonfinite"] == 0
                for u in arm["updates"]
            )
            eta = arm["eta"]
            curve = []
            for post in arm["post_updates"]:
                m = post["metrics"]
                if eta == 1:
                    assert "distill/anchor_kl" not in m
                    assert m["distill/teacher_kl"] == m["distill/target_kl"] == m["distill/objective"]
                    metric = "distill/full_kl"
                else:
                    check = (
                        eta * m["distill/teacher_kl"] + (1 - eta) * m["distill/anchor_kl"] + m["distill/target_log_z"]
                    )
                    identity = max(identity, abs(check - m["distill/target_kl"]))
                    metric = "distill/target_kl"
                if post["update"] < arm["steps"]:
                    alignment = max(alignment, abs(m["distill/target_kl"] - arm["updates"][post["update"]][metric]))
                assert abs(m["distill/target_log_z"] - arm["before"]["distill/target_log_z"]) < 1e-12
                curve.append(
                    [
                        post["update"],
                        m["distill/target_kl"],
                        m["distill/teacher_kl"],
                        m.get("distill/anchor_kl"),
                        m["distill/objective"],
                        m.get("distill/canonical_teacher_kl"),
                    ]
                )
            values = [arm["before"]["distill/teacher_kl"]] + [row[2] for row in curve]
            arms.append(
                {
                    "run": run,
                    "index": arm["index"],
                    "solver": arm["solver"],
                    "precision": arm["precision"],
                    "lr": arm["lr"],
                    "eta": eta,
                    "rounded": arm["rounded"],
                    "steps": arm["steps"],
                    "before": arm["before"],
                    "after": arm["after"],
                    "best": arm["best"],
                    "teacher_upward_steps": sum(b - a > 1e-10 for a, b in zip(values[:-1], values[1:], strict=True)),
                    "max_grad_norm_preclip": max(u["optim/actor/grad_norm_preclip"] for u in arm["updates"]),
                    "post_series": curve,
                    "saved_checkpoint": arm.get("saved_checkpoint"),
                    "actor_rope": arm["actor_rope"],
                }
            )
    checkpoint_source = next(a for a in arms if a["run"] == "adam-confirm" and a["index"] == 0)
    reloaded = next(a for a in arms if a["run"] == "opd-continuation" and a["index"] == 0)
    assert reloaded["before"]["distill/teacher_kl"] == checkpoint_source["after"]["distill/teacher_kl"]
    assert reloaded["before"]["distill/teacher_kl"] == reloaded["after"]["distill/teacher_kl"]
    assert protocols["teacher-native-fixed"]["teacher_equals_canonical"]
    direct = json.loads((args.root / "teacher-rope.json").read_text())
    assert len(direct["rows"]) == 20
    assert all(
        row["exact_canonical"]
        for row in direct["rows"]
        if row["mode"] in ("canonical", "restore_canonical", "parameter_only_cast")
    )
    assert all(row["exact_rounded"] for row in direct["rows"] if row["mode"] == "upcast_rounded")
    distributed = json.loads((args.root / "buffer-fsdp.json").read_text())
    assert distributed["teacher_actor_hidden_exact"] and distributed["buffer_dtype"] == "float32"
    training_checks = {}
    for name in ("full-kl-fsdp", "full-kl-qwen3-fsdp"):
        check = json.loads((args.root / f"{name}.json").read_text())
        fields = ("checkpoint_roundtrip", "scheduler_optimizer_roundtrip", "resumed_next_update_equivalent")
        assert all(check[field] for field in fields)
        training_checks[name] = {
            "model_type": check["model_type"],
            "world_size": check["world_size"],
            **{field: check[field] for field in fields},
        }
    result = {
        "scope": "Fixed-prefix full-vocabulary reverse-KL OPD; not sampled OPD or task accuracy/throughput.",
        "baseline_commit": "f712af223",
        "protocols": protocols,
        "arms": arms,
        "post_series_columns": ["update", "target_kl", "teacher_kl", "anchor_kl", "objective", "canonical_teacher_kl"],
        "teacher_buffer_probe": direct,
        "fsdp_buffer_check": distributed,
        "fsdp_training_checks": training_checks,
        "checks": {
            "optimizer_calls": sum(a["steps"] for a in arms),
            "nonzero_lr_updates": sum(a["steps"] for a in arms if a["lr"] > 0),
            "native_gold_alignment_max_abs": alignment,
            "kl_identity_max_abs": identity,
            "checkpoint_score_exact": True,
        },
    }
    finite(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["checks"], indent=2))


if __name__ == "__main__":
    main()
