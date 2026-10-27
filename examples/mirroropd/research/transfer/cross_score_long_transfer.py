#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only parameter x rollout-source scoring at the single-pass endpoints."""

import argparse
import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from ..common import native as core
from ..common.metrics import fixed_metrics as metrics


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    for key in ["models", "out", "cache"]:
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--teacher", choices=["justrl", "polaris"], required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    assert (a.out / a.teacher / "complete.json").exists()
    cases = {s: core.read(a.out / a.teacher / f"fresh-{b:02d}.json") for s, b in [("initial", 0), ("final", 20)]}
    assert all(len(rows) == 64 for rows in cases.values())
    q = core.load(a, a.teacher)
    head = q.lm_head.weight.float()
    hidden = {}
    for source, rows in cases.items():
        for i, r in enumerate(rows):
            hidden[source, i] = core.hidden(q, r["prompt_ids"] + r["response_ids"][:-1])[
                len(r["prompt_ids"]) - 1 :
            ].detach()
    q.cpu()
    torch.cuda.empty_cache()
    output = []
    for parameter_stage in ["initial", "final"]:
        path = (
            a.models / core.NAMES["student"]
            if parameter_stage == "initial"
            else a.cache / "checkpoints" / a.teacher / "batch-20"
        )
        p = (
            AutoModelForCausalLM.from_pretrained(
                path, dtype=torch.float32, attn_implementation="flash_attention_2", local_files_only=True
            )
            .cuda()
            .eval()
        )
        for source, rows in cases.items():
            records = []
            for i, r in enumerate(rows):
                hp = core.hidden(p, r["prompt_ids"] + r["response_ids"][:-1])[len(r["prompt_ids"]) - 1 :]
                hq = hidden[source, i]
                sums = {}
                for start in range(0, len(hp), 128):
                    sp = torch.nn.functional.linear(hp[start : start + 128].float(), p.lm_head.weight)
                    tq = torch.nn.functional.linear(hq[start : start + 128].float(), head)
                    for key, value in metrics(sp, tq).items():
                        sums[key] = sums.get(key, 0.0) + float(value.double().sum())
                records.append(dict(id=r["id"], seed=r["seed"], length=len(hp), sums=sums))
                del hp, hq, sp, tq
            output.append(
                dict(
                    teacher=a.teacher,
                    parameters=parameter_stage,
                    states=source,
                    records=records,
                    sequence_rkl=sum(r["sums"]["rkl"] for r in records) / 64,
                )
            )
            core.dump(a.out / f"cross-score-{a.teacher}.json", output)
            print("CROSS", a.teacher, parameter_stage, source, output[-1]["sequence_rkl"], flush=True)
        del p
        gc.collect()
        torch.cuda.empty_cache()
    core.dump(a.out / f"cross-score-{a.teacher}-complete.json", dict(finished=True, cases=4))


if __name__ == "__main__":
    main()
