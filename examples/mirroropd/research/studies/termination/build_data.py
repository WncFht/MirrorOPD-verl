# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Recompute all figure tables using only the archived data beside this script.

Semantic ledger: one response is one rollout; one update is one pre-optimizer
observation; position bins use actual response length and sum/count aggregation.
No smoothing, resampling, missing-value filling, or cross-objective relabeling.
"""

import argparse
import csv
import gzip
import json
import statistics as st
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from examples.mirroropd.research.core.io import dump
from examples.mirroropd.research.core.paths import MIRROROPD

ROOT = MIRROROPD / "docs/infra/model-compatibility/termination"
DATA = ROOT / "data"
ARMS = ("baseline", "aligned")
LABELS = {"baseline": "原始词表 KL", "aligned": "终止动作对齐"}
BINS = [f"p{i:02d}_{i + 10:02d}" for i in range(0, 100, 10)]


def load(path):
    return json.loads(path.read_text())


def jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def csvfile(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def repeat_metrics(ids):
    n = len(ids)
    grams = max(0, n - 7)
    unique = len(set(zip(*(ids[i:] for i in range(8)), strict=False))) if grams else 0
    a = np.asarray(ids, dtype=np.int64)
    span = 0
    period_best = None
    for period in range(1, min(128, n // 4) + 1):
        eq = a[period:] == a[:-period]
        edge = np.diff(np.concatenate(([False], eq, [False])).astype(np.int8))
        starts = np.flatnonzero(edge == 1)
        ends = np.flatnonzero(edge == -1)
        if len(starts):
            run = int((ends - starts).max())
            if run >= 3 * period and run + period >= 128 and run + period > span:
                span = run + period
                period_best = period
    return {
        "length": n,
        "cap_hit": n >= 7168,
        "long_loop": span >= 128,
        "repeat8": 1 - unique / grams if grams else 0.0,
        "loop_span": span,
        "loop_period": period_best,
    }


def summarize(responses):
    lengths = [r["length"] for r in responses]
    return {
        "responses": len(responses),
        "mean_length": st.mean(lengths),
        "median_length": st.median(lengths),
        "p95_length": float(np.percentile(lengths, 95)),
        "cap_pct": 100 * st.mean(r["cap_hit"] for r in responses),
        "loop_pct": 100 * st.mean(r["long_loop"] for r in responses),
        "repeat8_pct": 100 * st.mean(r["repeat8"] for r in responses),
        "short_pct": 100 * st.mean(r["length"] < 32 for r in responses),
        "score_pct": 100 * st.mean(r["score"] for r in responses),
        "stop_pct": 100 * st.mean(r["last_token"] in (151643, 151645) for r in responses),
    }


def main():
    global DATA, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    DATA, OUT = args.data.resolve(), args.out.resolve()
    if DATA == OUT:
        parser.error("Use a separate output directory; frozen evidence is read-only")
    OUT.mkdir(parents=True, exist_ok=True)
    steps = []
    updates = []
    positions = []
    responses = []
    windows = []
    profiles = []
    reuse = []
    gpu_summary = {}
    originals = {}
    checks = {}
    for arm in ARMS:
        logs = [r["data"] for r in jsonl(DATA / arm / "metrics.jsonl.gz")]
        raw = jsonl(DATA / arm / "responses.jsonl.gz")
        originals[arm] = raw
        u = [r for r in logs if "optim/actor/update_applied" in r]
        outer = {int(r["step/outer"]): r for r in logs if "perf/timing/outer_step_seconds" in r}
        assert len(u) == 320 and [r["step/update"] for r in u] == list(range(1, 321))
        assert [r["step/optimizer"] for r in u] == list(range(1, 321)) and all(
            r["optim/actor/update_applied"] == 1 for r in u
        )
        assert set(outer) == set(range(1, 21)) and len(raw) == 1280
        assert not any(any(k.startswith("eval/") for k in r) for r in logs)
        by_step = defaultdict(list)
        by_update = defaultdict(list)
        position_error = []
        for r in raw:
            ids = r["response_token_ids"]
            assert len(ids) == r["response_length"]
            rec = {
                "arm": arm,
                "outer": r["step"],
                "uid": r["uid"],
                **repeat_metrics(ids),
                "last_token": ids[-1] if ids else None,
                "score": r["score"],
            }
            responses.append(rec)
            by_step[r["step"]].append(rec)
        for r in u:
            step = int(r["step/outer"])
            index = (int(r["step/update"]) - 1) % 16 + 1
            rec = {
                "arm": arm,
                "outer": step,
                "update": r["step/update"],
                "reuse_index": index,
                "raw_kl": r["distill/full_kl"],
                "optimization_loss": r["optim/actor/loss"],
                "grad_norm": r["optim/actor/grad_norm_preclip"],
                "eos_identity_kl": r.get("distill/eos_identity_kl"),
                "update_applied": r["optim/actor/update_applied"],
            }
            assert np.isfinite([rec["raw_kl"], rec["optimization_loss"], rec["grad_norm"]]).all()
            if arm == "aligned":
                assert abs(rec["optimization_loss"] - r["distill/termination_equiv_kl"]) < 1e-4
                assert abs(rec["raw_kl"] - r["distill/termination_equiv_kl"] - rec["eos_identity_kl"]) < 1e-4
            updates.append(rec)
            by_update[step].append(rec)
            s = n = 0.0
            for b, label in enumerate(BINS):
                stem = f"distill/full_kl_position/{label}"
                total = r[stem + "/sum"]
                count = r[stem + "/token_count"]
                positions.append(
                    {
                        "arm": arm,
                        "outer": step,
                        "update": r["step/update"],
                        "reuse_index": index,
                        "bin": b,
                        "sum": total,
                        "count": count,
                        "mean": total / count if count else None,
                    }
                )
                s += total
                n += count
            assert n == sum(x["length"] for x in by_step[step])
            position_error.append(abs(s / n - rec["raw_kl"]))
        assert max(position_error) < 1e-4
        reference = load(DATA / arm / "analysis-reference.json")
        for step in range(1, 21):
            group = by_step[step]
            batch = by_update[step]
            assert len(group) == 64 and len(batch) == 16
            metrics = summarize(group)
            ref = reference["batches"][step - 1]
            for k in ("mean_length", "median_length", "p95_length", "cap_pct", "loop_pct", "repeat8_pct", "short_pct"):
                assert abs(metrics[k] - ref[k]) < 1e-8, (arm, step, k)
            current = outer[step]
            assert abs(metrics["mean_length"] - current["rollout/response_length/mean"]) < 1e-8
            steps.append(
                {
                    "arm": arm,
                    "outer": step,
                    **metrics,
                    "kl_first": batch[0]["raw_kl"],
                    "kl_last": batch[-1]["raw_kl"],
                    "loss_first": batch[0]["optimization_loss"],
                    "loss_last": batch[-1]["optimization_loss"],
                    "outer_seconds": current["perf/timing/outer_step_seconds"],
                    "rollout_seconds": current["perf/timing/rollout_pipeline_seconds"],
                    "teacher_seconds": current["perf/timing/teacher_forward_seconds"],
                    "actor_seconds": current["perf/timing/update_actor_seconds"],
                }
            )
        for window, lo, hi in [("early", 1, 5), ("late", 16, 20)]:
            windows.append(
                {
                    "arm": arm,
                    "window": window,
                    **summarize([r for r in responses if r["arm"] == arm and lo <= r["outer"] <= hi]),
                }
            )
            for reuse_index in range(1, 17):
                group = [
                    r for r in updates if r["arm"] == arm and lo <= r["outer"] <= hi and r["reuse_index"] == reuse_index
                ]
                reuse.append(
                    {
                        "arm": arm,
                        "window": window,
                        "reuse_index": reuse_index,
                        "raw_kl": st.mean(r["raw_kl"] for r in group),
                        "optimization_loss": st.mean(r["optimization_loss"] for r in group),
                    }
                )
            for phase in (1, 16):
                for b in range(10):
                    group = [
                        r
                        for r in positions
                        if r["arm"] == arm and lo <= r["outer"] <= hi and r["reuse_index"] == phase and r["bin"] == b
                    ]
                    s = sum(r["sum"] for r in group)
                    n = sum(r["count"] for r in group)
                    profiles.append(
                        {
                            "arm": arm,
                            "window": window,
                            "reuse_index": phase,
                            "bin": b,
                            "sum": s,
                            "count": n,
                            "mean": s / n if n else None,
                        }
                    )
        gpu = jsonl(DATA / arm / "gpu-samples.jsonl.gz")
        gpu_summary[arm] = {
            str(i): max(g["used_gib"] for row in gpu for g in row["gpus"] if g["index"] == i) for i in range(4)
        }
        checks[arm] = {
            "updates": len(u),
            "responses": len(raw),
            "outer_steps": len(outer),
            "max_position_reconstruction_error": max(position_error),
            "actual_updates": sum(r["optim/actor/update_applied"] for r in u),
            "no_validation": True,
        }
    matched = []
    pairs = []
    for step in range(1, 21):
        left = [r for r in originals["baseline"] if r["step"] == step]
        right = [r for r in originals["aligned"] if r["step"] == step]
        assert sorted(r["input"] for r in left) == sorted(r["input"] for r in right)
        matched.append(step)
        by_input = defaultdict(deque)
        for r in right:
            by_input[r["input"]].append(r)
        lookup = {r["uid"]: r for r in responses if r["arm"] == "baseline" and r["outer"] == step}
        lookup_right = {r["uid"]: r for r in responses if r["arm"] == "aligned" and r["outer"] == step}
        for a in left:
            b = by_input[a["input"]].popleft()
            if step >= 16 and lookup[a["uid"]]["long_loop"] and not lookup_right[b["uid"]]["long_loop"]:
                pairs.append(
                    {
                        "outer": step,
                        "prompt": a["input"],
                        "ground_truth": a["gts"],
                        "baseline": {"score": a["score"], "length": a["response_length"], "ending": a["output"][-700:]},
                        "aligned": {"score": b["score"], "length": b["response_length"], "ending": b["output"][-700:]},
                    }
                )
    samples = []
    for correct in (True, False):
        pool = [
            p
            for p in pairs
            if bool(p["baseline"]["score"] > 0) == correct and bool(p["aligned"]["score"] > 0) == correct
        ]
        if pool:
            samples.append(sorted(pool, key=lambda p: (p["outer"], p["prompt"]))[0])
    probes = {arm: load(DATA / arm / "prefix-results.json") for arm in ARMS}
    stops = []
    loop_prefix = []
    for model, arm, student in [
        ("initial", "baseline", "initial"),
        ("baseline", "baseline", "trained"),
        ("aligned", "aligned", "trained"),
        ("teacher", "baseline", "initial"),
    ]:
        subset = [
            r
            for r in probes[arm]
            if r["category"] == "early_terminal" and r["label"] == "last_observed_token" and r["student"] == student
        ]
        assert len(subset) == 10
        for r in subset:
            stops.append(
                {
                    "model": model,
                    "uid": r["uid"],
                    "probability": r["teacher_stop"] if model == "teacher" else r["student_stop"],
                }
            )
    for cycles in (0, 1, 4, 16, 64):
        for arm in ARMS:
            subset = [
                r
                for r in probes[arm]
                if r["category"] == "late_loop" and r["label"] == f"loop_copies_{cycles}" and r["student"] == "trained"
            ]
            assert len(subset) == 10
            loop_prefix.append(
                {
                    "arm": arm,
                    "cycles": cycles,
                    "n": len(subset),
                    "teacher_next_median": st.median(r["teacher_next"] for r in subset),
                    "student_next_median": st.median(r["student_next"] for r in subset),
                    "raw_kl_median": st.median(r["kl"] for r in subset),
                }
            )
    summary = {
        "labels": LABELS,
        "steps": steps,
        "updates": updates,
        "windows": windows,
        "positions": positions,
        "profiles": profiles,
        "reuse": reuse,
        "responses": responses,
        "stops": stops,
        "loop_prefix": loop_prefix,
        "gpu_peak_gib": gpu_summary,
        "configuration": load(DATA / "configuration.json"),
    }
    dump(OUT / "figure-data.json", summary)
    dump(OUT / "sample-pairs.json", samples)
    dump(
        OUT / "audit.json",
        {
            "checks": checks,
            "matched_prompt_batches": matched,
            "position_rows": len(positions),
            "response_rows": len(responses),
            "source_scope": "All reconstruction uses the compressed raw archives in this directory",
            "passed": True,
        },
    )
    for name, rs in [
        ("outer-metrics.csv", steps),
        ("update-metrics.csv", updates),
        ("full-kl-position.csv", positions),
        ("position-profiles.csv", profiles),
        ("window-metrics.csv", windows),
        ("reuse-profiles.csv", reuse),
        ("response-metrics.csv", responses),
        ("fixed-prefix-stop.csv", stops),
        ("loop-prefix.csv", loop_prefix),
    ]:
        csvfile(name, rs)
    print(
        f"PASS: {len(responses)} responses, {len(updates)} updates, {len(positions)} position rows; "
        "prompts matched in all 20 batches"
    )


if __name__ == "__main__":
    main()
