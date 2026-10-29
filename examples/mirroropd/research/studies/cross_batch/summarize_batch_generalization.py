#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Separate acquisition, untrained-batch transfer, forgetting and domain transfer."""

import argparse
import collections
import json
import statistics as st
from functools import partial
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.core.statistics import mean_ci


def index(rows):
    return {r["id"]: r for r in rows}


def risk(r, k="rkl", band="all"):
    return r["stats"][band][k] / 1024


ci = partial(mean_ci, n=1000)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", type=Path, required=True)
    args = a.parse_args()
    p = args.data
    summary = []
    influences = []
    fresh = []
    health = []
    for folder in sorted(p.iterdir()):
        if not folder.is_dir() or not (folder / "protocol.json").exists() or not (folder / "training.json").exists():
            continue
        cfg = read(folder / "protocol.json")
        teacher = cfg["teacher"]
        schedule = cfg["schedule"]
        arm = folder.name
        warm = cfg.get("initial", "original student") != "original student"
        baseline = p / (f"baseline-warm-{teacher}" if warm else f"baseline-{teacher}")
        if not (baseline / "done").exists():
            continue
        base = index(read(baseline / "fixed.json"))
        groups = {g: [i for i, r in base.items() if r["group"] == g] for g in list("ABCDEFGH") + ["Code", "IF"]}
        groups["train_pool"] = sum([groups[g] for g in "ABCD"], [])
        groups["held_math"] = sum([groups[g] for g in "EFGH"], [])
        previous = base
        previous_step = 0
        for f in sorted(folder.glob("fixed-*.json"), key=lambda f: int(f.stem.split("-")[-1])):
            rows = read(f)
            now = index(rows)
            step = rows[0]["step"]
            assert now.keys() == base.keys()
            for group, ids in groups.items():
                entry = dict(
                    arm=arm,
                    teacher=teacher,
                    schedule=schedule,
                    step=step,
                    group=group,
                    prompts=len(ids),
                    direction=cfg.get("direction", "reverse"),
                    lr=cfg.get("lr", 1e-6),
                    rich=cfg.get("rich", False),
                    evaluation_scope="common64_subset" if group == "train_pool" and cfg.get("rich") else group,
                    warm=warm,
                )
                for metric in ["rkl", "fkl_shared", "js", "tv", "hp", "hq"]:
                    b = [risk(base[i], metric) for i in ids]
                    c = [risk(now[i], metric) for i in ids]
                    entry[metric + "_before"] = st.mean(b)
                    entry[metric + "_after"] = st.mean(c)
                    entry[metric + "_drop"] = 1 - st.mean(c) / st.mean(b) if st.mean(b) else None
                entry["rkl_absolute_drop_ci"] = ci([risk(base[i]) - risk(now[i]) for i in ids])
                entry["improved_prompt_fraction"] = st.mean(risk(now[i]) < risk(base[i]) for i in ids)
                summary.append(entry)
                before = st.mean(risk(previous[i]) for i in ids)
                after = st.mean(risk(now[i]) for i in ids)
                trained = (
                    ("A" if schedule == "single" else "ABCD"[(step // 16 - 1) % 4])
                    if schedule in ["single", "blocked"]
                    else ("rich_pool" if cfg.get("rich") else "ABCD")
                )
                influences.append(
                    dict(
                        arm=arm,
                        step_before=previous_step,
                        step_after=step,
                        trained_group=trained,
                        evaluated_group=group,
                        before=before,
                        after=after,
                        relative_drop=1 - after / before if before else None,
                        absolute_drop=before - after,
                    )
                )
            previous = now
            previous_step = step
        f = folder / "training.json"
        if f.exists():
            tr = read(f)
            visits = collections.Counter(i for r in tr for i in r["prompt_ids"])
            health.append(
                dict(
                    arm=arm,
                    updates=len(tr),
                    token_visits=sum(r["tokens"] for r in tr),
                    normalizer_sum=sum(r["normalizer"] for r in tr),
                    clip_count=sum(r["grad_norm"] > 1 for r in tr),
                    final_training_loss=tr[-1]["loss"],
                    visits=dict(visits),
                    mean_measured_parameter_step=st.mean(
                        r["parameter_step_norm"] for r in tr if "parameter_step_norm" in r
                    ),
                )
            )
        for f in folder.glob("fresh-*.json"):
            rows = read(f)
            if len(rows) != 64:
                continue
            old = read(baseline / "fresh.json")
            b = collections.defaultdict(list)
            c = collections.defaultdict(list)
            for r in old:
                b[r["id"]].append(r["stats"]["all"]["rkl"])
            for r in rows:
                c[r["id"]].append(r["stats"]["all"]["rkl"])
            deltas = [st.mean(b[i]) - st.mean(c[i]) for i in sorted(b)]
            fresh.append(
                dict(
                    arm=arm,
                    step=rows[0]["step"],
                    before=st.mean(st.mean(v) for v in b.values()),
                    after=st.mean(st.mean(v) for v in c.values()),
                    absolute_drop_ci=ci(deltas),
                    length=st.mean(len(r["response_ids"]) for r in rows),
                    stop_fraction=st.mean(r["finish_reason"] == "stop" for r in rows),
                )
            )
    dump(p / "transfer-summary.json", summary)
    dump(p / "block-influence.json", influences)
    dump(p / "fresh-summary.json", fresh)
    dump(p / "training-summary.json", health)
    for r in summary:
        if r["step"] in [16, 128] and r["group"] in ["A", "B", "train_pool", "held_math", "Code", "IF"]:
            print(
                r["arm"],
                r["step"],
                r["group"],
                round(r["rkl_before"], 5),
                "->",
                round(r["rkl_after"], 5),
                round(100 * r["rkl_drop"], 2),
            )
    print("fresh", json.dumps(fresh))


if __name__ == "__main__":
    main()
