# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit actual losses, disjoint prompt splits, frozen baselines and completion."""

import argparse
import math
import re
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    a = ap.parse_args()
    root = a.data
    checks = []

    def check(name, value, **details):
        checks.append(dict(check=name, passed=bool(value), **details))

    train = read(root / "reference-student.json")
    test = read(root / "test-bank.json")
    diag = read(root / "diagnostic-bank.json")
    check(
        "disjoint train/diagnostic/test prompt IDs",
        not (set(r["id"] for r in train) & set(r["id"] for r in test + diag)),
    )
    check("disjoint diagnostic/test prompt IDs", not (set(r["id"] for r in diag) & set(r["id"] for r in test)))
    check("fixed split sizes", (len(train), len(diag), len(test)) == (128, 64, 256))
    text_splits = {}
    for name, rows in [
        ("train", train),
        ("diagnostic", diag),
        ("test", test),
        ("task", read(root / "task-prompts.json")),
    ]:
        texts = {re.sub(r"\s+", " ", r["prompt"]).strip() for r in rows}
        check(name + ": unique normalized prompt text", len(texts) == len(rows))
        text_splits[name] = texts
    for i, name in enumerate(text_splits):
        for other in list(text_splits)[i + 1 :]:
            check(
                name + "/" + other + ": disjoint normalized prompt text",
                text_splits[name].isdisjoint(text_splits[other]),
            )
    baseline = {}
    states = []
    for cfg in read(root / "protocol.json")["arms"]:
        d = root / cfg["name"]
        f = d / "training.json"
        history = read(f) if f.exists() else []
        complete = (d / "complete.json").exists()
        states.append(cfg | dict(updates=len(history), complete=complete))
        if not (d / "fixed-000.json").exists():
            continue
        before = read(d / "fixed-000.json")
        check(cfg["name"] + ": frozen initial drift", max(abs(r["stats"]["all"]["retain_fkl"]) for r in before) < 1e-4)
        key = cfg["teacher"]
        indexed = {(r["view"], r["id"]): r for r in before}
        if key in baseline:
            error = max(
                abs(r["stats"]["all"][m] - baseline[key][k]["stats"]["all"][m]) / 1024
                for k, r in indexed.items()
                for m in ["js", "rkl", "fkl_shared", "hp", "reference_nll"]
            )
            check(cfg["name"] + ": shared baseline numerical equivalence", error < 1e-6, max_error=error)
        else:
            baseline[key] = indexed
        check(
            cfg["name"] + ": finite training records",
            all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"]) for r in history),
        )
        check(cfg["name"] + ": sequential update IDs", [r["step"] for r in history] == list(range(1, len(history) + 1)))
        check(
            cfg["name"] + ": disjoint batch order",
            all(
                r["batch"] == (r["step"] - 1) // 16 + 1
                and r["repeat"] == (r["step"] - 1) % 16 + 1
                and set(r["prompt_ids"]) == set(range((r["batch"] - 1) * 16, r["batch"] * 16))
                for r in history
            ),
        )
        metric = {"reverse": "rkl", "forward": "fkl_shared", "sft": "reference_nll"}[cfg["direction"]]
        errors = []
        for r in history:
            if r["repeat"] != 1:
                continue
            active = read(d / f"active-{r['batch']:02d}-before.json")
            expected = sum(v["stats"]["all"][metric] for v in active) / (16 * 1024)
            errors.append(abs(expected - r["loss"]))
        if errors:
            check(cfg["name"] + ": objective equals pre-update evaluation", max(errors) < 2e-6, max_error=max(errors))
        if cfg["source"] == "online" and (d / "rollout-01.json").exists():
            rollout = read(d / "rollout-01.json")
            check(
                cfg["name"] + ": online initial state paired with frozen student",
                [r["response_ids"] for r in rollout] == [r["response_ids"] for r in train[:16]],
            )
        if complete:
            check(
                cfg["name"] + ": 128 updates and eight batches",
                len(history) == 128 and len({r["batch"] for r in history}) == 8,
            )
            check(
                cfg["name"] + ": five fixed evaluations",
                all((d / f"fixed-{s:03d}.json").exists() for s in [0, 32, 64, 96, 128]),
            )
            task = read(d / "task.json")
            check(cfg["name"] + ": complete graded task", len(task) == 64 and all("grade" in r for r in task))
            check(
                cfg["name"] + ": fresh evaluations", all(len(read(d / f"fresh-{s:03d}.json")) == 64 for s in [64, 128])
            )
    synthetic_complete = 0
    synthetic_updates = 0
    for kind in ["shared", "random"]:
        for direction in ["reverse", "forward"]:
            d = root / f"synthetic-{kind}-{direction}"
            if not (d / "complete.json").exists():
                continue
            synthetic_complete += 1
            complete = read(d / "complete.json")
            history = read(d / "training.json")
            synthetic_updates += len(history)
            rows = read(d / "prompts.json")
            values = read(d / f"eval-{complete['steps']:03d}.json")
            check(
                d.name + ": full parameter finite training",
                len(history) == complete["steps"] <= 512
                and all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"]) for r in history),
            )
            check(
                d.name + ": disjoint train/test suffixes",
                len(rows) == 80 and {r["suffix"] for r in rows[:16]}.isdisjoint({r["suffix"] for r in rows[16:]}),
            )
            criterion = all(values["top1_correct"][:16]) and sum(values["correct_probability"][:16]) / 16 >= 0.8
            check(
                d.name + ": train-only stopping",
                complete["training_criterion_reached"] == criterion and (complete["steps"] == 512 or criterion),
            )
            check(
                d.name + ": labels fixed and balanced",
                all(
                    sorted(r["random_label"] for r in rows[:16] if r["prefix"] == prefix) == [1, 2, 3, 4]
                    for prefix in ["dax", "wug", "zup", "vem"]
                ),
            )
    functional_complete = 0
    for file in (root / "if-eval").glob("*.json"):
        rows = read(file)
        if len(rows) != 64 or not all("grade" in r for r in rows):
            continue
        functional_complete += 1
        expected = {r["id"]: r for r in test if r["domain"] == "if"}
        check(
            file.stem + ": IF prompt and sampling protocol",
            len(rows) == 64
            and all(
                r["prompt_ids"] == expected[r["id"]]["prompt_ids"]
                and r["cap"] == 4096
                and r["seed"] == 99700 + (i // 16) * 16
                and 0 < len(r["response_ids"]) <= 4096
                for i, r in enumerate(rows)
            ),
        )
    calibration_complete = 0
    for file in (root / "calibration").glob("*.json"):
        r = read(file)
        if not r["complete"]:
            continue
        calibration_complete += 1
        temperatures = r["protocol"]["temperatures"]
        chosen = min(temperatures, key=lambda t: r["risks"]["diagnostic"][str(t)]["js"])
        check(r["arm"] + ": diagnostic-only temperature selection", chosen == r["selected_temperature"])
        check(
            r["arm"] + ": calibration split identities",
            {v["id"] for v in r["rows"] if v["split"] == "diagnostic"} == {v["id"] for v in diag}
            and {v["id"] for v in r["rows"] if v["split"] == "test"}
            == {v["id"] for v in test if v["domain"] == "math"},
        )
    continuation_complete = 0
    for file in (root / "continuation").glob("*.json"):
        record = read(file)
        if not record["complete"]:
            continue
        continuation_complete += 1
        original = {r["id"]: r for r in read(root / record["arm"] / "task.json")}
        check(
            record["arm"] + ": continuation preserves complete sampled prefix",
            all(
                r["response_ids"][: r["original_response_length"]] == original[r["id"]]["response_ids"]
                and r["prompt_ids"] == original[r["id"]]["prompt_ids"]
                for r in record["rows"]
            ),
        )
        check(
            record["arm"] + ": continuation uses preregistered cases and budget",
            {r["id"] for r in record["rows"]} == set(record["protocol"]["arms"][record["arm"]]["ids"])
            and all(
                0 < len(r["suffix_ids"]) <= 8192
                and r["original_cap"] == 8192
                and r["cap"] == 16384
                and r["original_finish_reason"] == "length"
                and not r["original_grade"]["acc"]
                for r in record["rows"]
            ),
        )
    result = dict(
        checks=checks,
        states=states,
        complete_arms=sum(r["complete"] for r in states),
        expected_arms=14,
        synthetic_complete=synthetic_complete,
        expected_synthetic=4,
        functional_if_complete=functional_complete,
        expected_functional_if=17,
        calibration_complete=calibration_complete,
        expected_calibration=5,
        continuation_complete=continuation_complete,
        expected_continuation=2,
        all_observed_checks_pass=all(c["passed"] for c in checks),
        training_updates_total=sum(r["updates"] for r in states) + synthetic_updates,
    )
    result["pipeline_complete"] = (
        result["complete_arms"] == 14
        and synthetic_complete == 4
        and functional_complete == 17
        and calibration_complete == 5
        and continuation_complete == 2
        and result["all_observed_checks_pass"]
    )
    dump(root / "audit.json", result)
    print(
        "Completed",
        result["complete_arms"],
        f"/ 14; synthetic {synthetic_complete}/4; IF {functional_complete}/17; "
        f"calibration {calibration_complete}/5; continuation {continuation_complete}/2; observed checks",
        len(checks),
        "; failures",
        sum(not c["passed"] for c in checks),
    )
    for c in checks:
        if not c["passed"]:
            print(c)
    if not result["all_observed_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
