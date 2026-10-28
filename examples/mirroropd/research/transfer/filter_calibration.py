# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Post-training scalar-temperature diagnosis, selected on disjoint math prompts.

No parameter updates. The held-out set never selects a temperature. This tests
probability calibration on fixed prefixes, not functional task improvement.
"""

import argparse
import math
import statistics as st
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from ..common import native as core
from . import filter_common as fc


@torch.no_grad()
def run(a):
    target = a.out / "calibration" / (a.arm + ".json")
    assert not target.exists()
    protocol = core.read(a.out / "protocol.json")["posthoc_calibration"]
    temperatures = protocol["temperatures"]
    diagnostic = core.read(a.out / "diagnostic-bank.json")
    test = [r for r in core.read(a.out / "test-bank.json") if r["domain"] == "math"]
    teacher_test = [r for r in core.read(a.out / "reference-polaris.json") if r["role"] == "test"]
    assert len(teacher_test) == 64
    rows = diagnostic + test + teacher_test
    q = core.load(a, "polaris")
    wq = q.lm_head.weight.float()
    fc.cache_hidden(a.cache, "polaris", q, rows)
    q.cpu()
    torch.cuda.empty_cache()
    hq = fc.load_hidden(a.cache, "polaris", rows, model=q)
    if a.arm == "initial":
        p = core.load(a, "student", student=True)
    else:
        p = (
            AutoModelForCausalLM.from_pretrained(
                a.cache / "checkpoints" / a.arm / "step-128",
                dtype=torch.float32,
                attn_implementation="flash_attention_2",
                local_files_only=True,
            )
            .cuda()
            .eval()
        )
    p.requires_grad_(False)
    results = []
    for index, row in enumerate(rows):
        hp = core.hidden(p, row["prompt_ids"] + row["response_ids"][:-1])[len(row["prompt_ids"]) - 1 :]
        ht = hq[row["source"], row["id"]].cuda()
        sums = {str(t): dict(js=0.0, rkl=0.0, fkl=0.0, entropy=0.0) for t in temperatures}
        for start in range(0, len(hp), 128):
            zp = torch.nn.functional.linear(hp[start : start + 128].float(), p.lm_head.weight)
            zq = torch.nn.functional.linear(ht[start : start + 128].float(), wq)
            v = zp.shape[-1]
            lq = zq.log_softmax(-1)
            lqc = zq[:, :v].log_softmax(-1)
            for t in temperatures:
                lp = (zp / t).log_softmax(-1)
                mix = torch.logaddexp(lp, lq[:, :v]) - math.log(2)
                sums[str(t)]["js"] += float(
                    0.5
                    * (
                        (lp.exp() * (lp - mix)).sum()
                        + (lq[:, :v].exp() * (lq[:, :v] - mix)).sum()
                        + lq[:, v:].exp().sum() * math.log(2)
                    )
                )
                sums[str(t)]["rkl"] += float((lp.exp() * (lp - lq[:, :v])).sum())
                sums[str(t)]["fkl"] += float((lqc.exp() * (lqc - lp)).sum())
                sums[str(t)]["entropy"] += float(-(lp.exp() * lp).sum())
        split = "diagnostic" if index < 64 else ("test" if index < 192 else "teacher_test")
        results.append(dict(id=row["id"], source=row["source"], split=split, length=len(hp), scores=sums))
        del hp, ht, zp, zq, lp, lq, lqc, mix
        if (index + 1) % 32 == 0:
            print("CALIBRATION", a.arm, index + 1, "/", len(rows), flush=True)
    risks = {}
    for split in ["diagnostic", "test", "teacher_test"]:
        rs = [r for r in results if r["split"] == split]
        risks[split] = {
            str(t): {k: st.mean(r["scores"][str(t)][k] / 1024 for r in rs) for k in ["js", "rkl", "fkl", "entropy"]}
            for t in temperatures
        }
    chosen = min(temperatures, key=lambda t: risks["diagnostic"][str(t)]["js"])
    core.dump(
        target,
        dict(
            arm=a.arm,
            protocol=protocol,
            selected_temperature=chosen,
            selection_at_grid_boundary=chosen in [min(temperatures), max(temperatures)],
            risks=risks,
            rows=results,
            complete=True,
        ),
    )
    print("CALIBRATION DONE", a.arm, "temperature", chosen, flush=True)


def main():
    ap = argparse.ArgumentParser()
    for key in ["models", "out", "cache"]:
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--arm", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    run(a)


if __name__ == "__main__":
    main()
