#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit top-16 cardinality, probability mass and nonnegative KL contributions.

Use common frozen E/F prefixes for both checkpoints. The per-token allocation
p log(p/q)-p+q is nonnegative and sums to native RKL on the union vocabulary.
"""

import argparse
from pathlib import Path

import torch

from examples.mirroropd.research.backends import hf_native as core
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.cross_batch.batch_size_stage import student


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    for name in ["models", "out", "cache", "previous-cache"]:
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--phase", choices=["initial", "warm128"], required=True)
    ap.add_argument("--state-source", choices=["bank", "fresh-initial", "fresh-warm128"], default="bank")
    a = ap.parse_args()
    a.teacher = "polaris"
    suffix = "" if a.state_source == "bank" else "-" + a.state_source
    destination = a.out / f"overlap-mass-{a.phase}{suffix}.json"
    if destination.exists():
        done = read(destination)
        assert done["phase"] == a.phase and done["models"] == str(a.models.resolve()) and done["prompts"] == 32
        print("Reusing completed frozen-prefix probe", destination, flush=True)
        return
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    p, q = student(a), core.load(a, "polaris")
    wq = q.lm_head.weight.float()
    records = []
    inputs = (
        read(a.out / "bank.json")
        if a.state_source == "bank"
        else read(a.out / f"baseline-polaris-{a.state_source.removeprefix('fresh-')}/fresh.json")
    )
    for r in inputs:
        if r["group"] not in ["E", "F"]:
            continue
        hp = core.hidden(p, r["prompt_ids"] + r["response_ids"][:-1])[len(r["prompt_ids"]) - 1 :]
        hq = core.teacher_hidden(a, q, r, cache=a.state_source == "bank")
        sums = {}
        count = 0
        for start in range(0, len(hp), 128):
            lp = torch.nn.functional.linear(hp[start : start + 128].float(), p.lm_head.weight).log_softmax(-1)
            lq = torch.nn.functional.linear(hq[start : start + 128].float(), wq).log_softmax(-1)
            prob = lp.exp()
            qprob = lq.exp()
            v = prob.shape[-1]
            si = lp.topk(16, -1).indices
            ti = lq.topk(16, -1).indices
            smask = torch.zeros_like(prob, dtype=torch.bool).scatter_(1, si, True)
            tmask = torch.zeros_like(qprob, dtype=torch.bool).scatter_(1, ti, True)
            category = 2 * smask.long() + tmask[:, :v].long()
            generalized = prob * (lp - lq[:, :v]) - prob + qprob[:, :v]
            vals = {
                "rkl": (prob * (lp - lq[:, :v])).sum(-1),
                "hp": -(prob * lp).sum(-1),
                "hq": -(qprob * lq).sum(-1),
                "overlap16": ((si[:, :, None] == ti[:, None, :]).any(-1)).float().mean(-1),
            }
            for c in range(4):
                mask = category == c
                vals[f"p_mass_{c}"] = (prob * mask).sum(-1)
                vals[f"q_mass_{c}"] = (qprob[:, :v] * mask).sum(-1)
                vals[f"gkl_{c}"] = (generalized * mask).sum(-1)
                if c in [0, 1]:
                    extra = (~tmask[:, v:] if c == 0 else tmask[:, v:]) * qprob[:, v:]
                    vals[f"q_mass_{c}"] += extra.sum(-1)
                    vals[f"gkl_{c}"] += extra.sum(-1)
            assert torch.allclose(sum(vals[f"gkl_{c}"] for c in range(4)), vals["rkl"], atol=3e-6, rtol=1e-4)
            for lo, hi in [(0, 1), (1, 4), (4, 8), (8, 16)]:
                ids = si[:, lo:hi]
                shared = (ids[:, :, None] == ti[:, None, :]).any(-1)
                vals[f"rank_{lo + 1}_{hi}_p_mass"] = prob.gather(1, ids).sum(-1)
                vals[f"rank_{lo + 1}_{hi}_common_p_mass"] = (prob.gather(1, ids) * shared).sum(-1)
                vals[f"rank_{lo + 1}_{hi}_shared_fraction"] = shared.float().mean(-1)
            count += len(lp)
            for k, val in vals.items():
                sums[k] = sums.get(k, 0.0) + float(val.double().sum())
        records.append(dict(id=r["id"], seed=r.get("seed"), tokens=count, sums=sums))
        print("rank", a.phase, r["id"], flush=True)
    tokens = sum(r["tokens"] for r in records)
    means = {k: sum(r["sums"][k] for r in records) / tokens for k in records[0]["sums"]}
    dump(
        destination,
        dict(
            phase=a.phase,
            state_source=a.state_source,
            models=str(a.models.resolve()),
            prompts=32,
            trajectories=len(records),
            tokens=tokens,
            means=means,
            records=records,
            definition=(
                "Frozen states from the specified source; four categories 0 neither, 1 teacher only, 2 st"
                "udent only, 3 intersection. GKL is a nonnegative additive allocation of full native RKL."
            ),
        ),
    )
    print(means, flush=True)


if __name__ == "__main__":
    main()
