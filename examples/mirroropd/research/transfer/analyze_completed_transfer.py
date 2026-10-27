#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Completed E48/E20 comparison from explicitly frozen local snapshots.

Windows are descriptive averages of different prompt/trajectory batches.
Reuse observations 1 and 16 bracket 15 optimizer operations.
"""

import argparse
import collections
import csv
import gzip
import json
import math
import statistics as st
from pathlib import Path

from ..common.io import dump


def read(path):
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), float | int) and math.isfinite(r[key])]
    return st.mean(vals) if vals else None


def run(rows):
    updates = [r for r in rows if "distill/full_kl" in r]
    outers = [r for r in rows if "rollout/response_tokens" in r]
    groups = collections.defaultdict(list)
    for r in updates:
        groups[r["step/outer"]].append(r)
    groups = {o: sorted(rs, key=lambda r: r["step/update"]) for o, rs in groups.items()}
    outer_by_step = {r["step/outer"]: r for r in outers}
    assert [r["step/update"] for r in updates] == list(range(1, 4097))
    assert all(len(v) == 16 for v in groups.values()) and len(outers) == 256
    batches = []
    for o, rs in sorted(groups.items()):
        kl = [r["distill/full_kl"] for r in rs]
        batches.append(
            dict(
                outer=o,
                first=kl[0],
                last=kl[-1],
                relative_drop=1 - kl[-1] / kl[0],
                absolute_drop=kl[0] - kl[-1],
                first_four_rate=(kl[0] - kl[4]) / 4,
                last_four_rate=(kl[11] - kl[15]) / 4,
                new_sequence_conditional_kl=kl[0]
                * outer_by_step[o]["rollout/response_tokens"]
                / outer_by_step[o]["rollout/response_count"],
                first_grad=rs[0]["optim/actor/grad_norm_preclip"],
                last_grad=rs[-1]["optim/actor/grad_norm_preclip"],
            )
        )
    windows = []
    for lo in range(1, 257, 32):
        hi = lo + 31
        bs = [b for b in batches if lo <= b["outer"] <= hi]
        os = [r for r in outers if lo <= r["step/outer"] <= hi]
        ends = [groups[o] for o in range(lo, hi + 1)]
        w = dict(lo=lo, hi=hi, **{k: mean(bs, k) for k in bs[0] if k != "outer"})
        w["outer_metrics"] = {k: mean(os, k) for k in os[0] if not k.startswith("_")}
        w["reuse_kl"] = [mean([v[i] for v in ends], "distill/full_kl") for i in range(16)]
        w["reuse_grad"] = [mean([v[i] for v in ends], "optim/actor/grad_norm_preclip") for i in range(16)]
        w["accelerating_batches"] = sum(b["last_four_rate"] > b["first_four_rate"] for b in bs)
        for side, idx in [("first", 0), ("last", 15)]:
            if "distill/full_kl_position/p00_10/sum" in ends[0][idx]:
                w["positions_" + side] = []
                for b in range(0, 100, 10):
                    stem = f"distill/full_kl_position/p{b:02d}_{b + 10:02d}/"
                    count = sum(v[idx][stem + "token_count"] for v in ends)
                    value = sum(v[idx][stem + "sum"] for v in ends)
                    w["positions_" + side].append(dict(sum=value, count=count, mean=value / count))
        windows.append(w)
    decomposition = []
    for lo, hi in [(1, 58), (1, 256), (65, 256), (193, 256), (225, 256)]:
        seq = [groups[o] for o in range(lo, hi + 1)]
        drops = sum(v[0]["distill/full_kl"] - v[-1]["distill/full_kl"] for v in seq[:-1])
        jumps = sum(b[0]["distill/full_kl"] - a[-1]["distill/full_kl"] for a, b in zip(seq, seq[1:], strict=False))
        net = seq[-1][0]["distill/full_kl"] - seq[0][0]["distill/full_kl"]
        assert abs(net - (jumps - drops)) < 1e-12
        decomposition.append(
            dict(
                lo=lo,
                hi=hi,
                within_drop_sum=drops,
                next_batch_jump_sum=jumps,
                jump_drop_ratio=jumps / drops,
                first_observation_change=net,
            )
        )
    audit = dict(
        history_rows=len(rows),
        updates=len(updates),
        outers=len(outers),
        metric_count=len({k for r in rows for k in r}),
        nonfinite_loss=sum(r["optim/actor/loss_nonfinite"] for r in updates),
        nonfinite_grad=sum(r["optim/actor/grad_norm_preclip_nonfinite"] for r in updates),
        skipped=sum(r["optim/actor/update_skipped"] for r in updates),
        grad_clip_count=sum(r["optim/actor/grad_norm_preclip"] > 1 for r in updates),
        clip_updates=[r["step/update"] for r in updates if r["optim/actor/grad_norm_preclip"] > 1],
        lrs=sorted({r["optim/actor/lr"] for r in updates}),
        inner_decreases=sum(
            b["distill/full_kl"] < a["distill/full_kl"]
            for rs in groups.values()
            for a, b in zip(rs, rs[1:], strict=False)
        ),
        inner_pairs=256 * 15,
        loss_kl_max_error=max(abs(r["optim/actor/loss"] - r["distill/full_kl"]) for r in updates),
        response_tokens=sum(r["rollout/response_tokens"] for r in outers),
        prompt_slots=sum(r["rollout/response_count"] for r in outers),
        outer_seconds=sum(r["perf/timing/outer_step_seconds"] for r in outers),
        validation_seconds=sum(r.get("perf/timing/validation_seconds", 0) for r in rows),
        checkpoint_seconds=sum(r.get("perf/timing/save_checkpoint_seconds", 0) for r in rows),
        aborted=sum(r["rollout/finish_reason/aborted_count"] for r in outers),
        max_empty_loss_mask=max(r["rollout/empty_loss_mask_fraction"] for r in outers),
    )
    if "distill/full_kl_position/p00_10/sum" in updates[0]:
        audit["position_identity_max_error"] = max(
            abs(
                r["distill/full_kl"]
                - sum(r[f"distill/full_kl_position/p{b:02d}_{b + 10:02d}/sum"] for b in range(0, 100, 10))
                / sum(r[f"distill/full_kl_position/p{b:02d}_{b + 10:02d}/token_count"] for b in range(0, 100, 10))
            )
            for r in updates
        )
    evals = [
        {k: v for k, v in r.items() if k.startswith(("step/", "eval/"))} for r in rows if "eval/math500/acc/mean" in r
    ]
    return dict(audit=audit, batches=batches, windows=windows, decomposition=decomposition, evals=evals)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--old", type=Path, required=True)
    args = a.parse_args()
    allstats = {}
    for label, folder in [("E48", "run"), ("E20", "E20")]:
        rows = read(args.data / folder / "history.jsonl")
        allstats[label] = run(rows)
        if label == "E48":
            previous = read(args.old / "run/history.jsonl")
            assert rows[: len(previous)] == previous
            assert json.loads((args.data / folder / "config.json").read_text()) == json.loads(
                (args.old / "run/config.json").read_text()
            )
            allstats[label]["audit"]["old_history_prefix_identical"] = True
            allstats[label]["audit"]["old_config_identical"] = True
        inventory = []
        for k in sorted({k for r in rows for k in r}):
            vals = [r[k] for r in rows if isinstance(r.get(k), int | float) and math.isfinite(r[k])]
            if vals:
                inventory.append(
                    dict(
                        metric=k,
                        n=len(vals),
                        first=vals[0],
                        last=vals[-1],
                        min=min(vals),
                        max=max(vals),
                        mean=st.mean(vals),
                    )
                )
        with (args.data / f"{label}-all-metrics.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(inventory[0]))
            w.writeheader()
            w.writerows(inventory)
        system = read(args.data / folder / "system.jsonl.gz")
        allstats[label]["system_summary"] = {}
        for suffix in ["gpu", "memoryAllocatedBytes", "powerWatts", "temp", "uncorrectedMemoryErrors"]:
            d = {}
            for gpu in range(8):
                key = f"system.gpu.{gpu}.{suffix}"
                vals = [r[key] for r in system if key in r]
                if vals:
                    d[str(gpu)] = dict(n=len(vals), mean=st.mean(vals), max=max(vals))
            allstats[label]["system_summary"][suffix] = d
    dump(args.data / "statistics.json", allstats)
    print(json.dumps({k: v["audit"] for k, v in allstats.items()}, indent=2))


if __name__ == "__main__":
    main()
