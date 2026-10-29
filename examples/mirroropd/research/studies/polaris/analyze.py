#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Pair every post-update state with its identical baseline and evaluation view."""

import argparse
import collections
import random
import statistics as st
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read


def mean(rs, k):
    return st.mean(r[k] for r in rs)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", type=Path, required=True)
    args = a.parse_args()
    root = args.data
    summary = []
    diagnostics = []
    for folder in sorted(root.glob("*-fold*")):
        if not all((folder / name).exists() for name in ["protocol.json", "training.json", "evaluation.json"]):
            continue
        cfg = read(folder / "protocol.json")
        hist = read(folder / "training.json")
        rows = read(folder / "evaluation.json")
        base = {(r["uid"], r["view"], r["region"]): r for r in rows if r["step"] == 0}
        groups = collections.defaultdict(list)
        for r in rows:
            if r["step"] > 0:
                part = "test" if r["role"] == "test" else ("batch_A" if r["id"] in cfg["train_ids"][:6] else "batch_B")
                groups[(r["step"], r["view"], part, r["source"], r["domain"], r["region"])].append(r)
        for key, rs in groups.items():
            before = [base[(r["uid"], r["view"], r["region"])] for r in rs]
            delta = [b["rkl"] - r["rkl"] for b, r in zip(before, rs, strict=False)]
            rng = random.Random(42)
            ci = None
            if key[0] == cfg["steps"] and key[2] == "test" and key[-1] == "all":
                bs = sorted(sum(rng.choices(delta, k=len(delta))) / len(delta) for _ in range(1000))
                ci = [bs[24], bs[974]]
            rec = dict(
                arm=folder.name,
                initial=cfg["initial"],
                optimizer=cfg["optimizer"],
                training_precision=cfg["precision"],
                fold=int(folder.name.split("-fold")[1].split("-")[0]),
                accum_fp32=cfg.get("accum_fp32", False),
                calibrated=cfg["calibration"] is not None,
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
                **dict(zip(["step", "view", "partition", "source", "domain", "region"], key, strict=False)),
                prompts=len(rs),
                rkl_before=mean(before, "rkl"),
                rkl_after=mean(rs, "rkl"),
                relative_drop=1 - mean(rs, "rkl") / mean(before, "rkl"),
                paired_absolute_drop_ci=ci,
                js_before=mean(before, "js"),
                js_after=mean(rs, "js"),
                tv_before=mean(before, "tv"),
                tv_after=mean(rs, "tv"),
                drift=mean(rs, "drift"),
                improved_fraction=st.mean(v > 0 for v in delta),
            )
            summary.append(rec)
        diagnostics.append(
            dict(
                arm=folder.name,
                updates=len(hist),
                complete=(folder / "complete.json").exists(),
                clipped=sum(r["grad_norm"] > 1 for r in hist),
                first_step=hist[0],
                last_step=hist[-1],
                step_l2_mean=mean(hist, "step_l2"),
                visible_fraction_mean=mean(hist, "bf16_visible_fraction"),
            )
        )
    dump(root / "summary.json", summary)
    dump(root / "optimization-summary.json", diagnostics)
    for folder in sorted({r["arm"] for r in summary}):
        rows = [r for r in summary if r["arm"] == folder]
        end = max(r["step"] for r in rows)
        selected = [
            r
            for r in rows
            if r["step"] == end
            and r["partition"] == "test"
            and r["source"] == "student"
            and r["region"] == "all"
            and r["domain"] == "math"
        ]
        print(
            folder,
            "step",
            end,
            [
                (r["view"], round(r["rkl_after"], 6), round(100 * r["relative_drop"], 2), round(r["drift"], 6))
                for r in selected
            ],
        )


if __name__ == "__main__":
    main()
