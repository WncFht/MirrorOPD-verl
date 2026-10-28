#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Pre-training reference-JS gradient alignment; disjoint diagnostic and test banks."""

import argparse
import math
from pathlib import Path

import torch

from ..common import native as core
from . import filter_common as fc


def gradient(p, wq, hq, rows):
    p.zero_grad(set_to_none=True)
    value = 0.0
    for r in rows:
        hp = core.hidden(p, r["prompt_ids"] + r["response_ids"][:-1])[len(r["prompt_ids"]) - 1 :]
        ht = hq[r["source"], r["id"]].cuda()
        terms = []
        for start in range(0, len(hp), 128):
            lp = torch.nn.functional.linear(hp[start : start + 128].float(), p.lm_head.weight).log_softmax(-1)
            with torch.no_grad():
                lq = torch.nn.functional.linear(ht[start : start + 128].float(), wq).log_softmax(-1)
            q = lq.exp()
            v = lp.shape[-1]
            mix = torch.logaddexp(lp, lq[:, :v]) - math.log(2)
            terms.append(
                0.5
                * ((lp.exp() * (lp - mix)).sum() + (q[:, :v] * (lq[:, :v] - mix)).sum() + q[:, v:].sum() * math.log(2))
                / (len(rows) * 1024)
            )
        loss = sum(terms)
        loss.backward()
        value += float(loss.detach())
        del hp, ht, lp, lq, q, mix, loss, terms
    flat = torch.cat(
        [(x.grad.detach() if x.grad is not None else torch.zeros_like(x)).flatten() for x in p.parameters()]
    )
    p.zero_grad(set_to_none=True)
    return flat, value


def products(vec, refs):
    values = torch.zeros(len(refs) + 1, device="cuda", dtype=torch.float64)
    for start in range(0, len(vec), 2**22):
        part = vec[start : start + 2**22]
        values[0] += (part * part).sum(dtype=torch.float64)
        for i, r in enumerate(refs, 1):
            values[i] += (part * r[start : start + 2**22]).sum(dtype=torch.float64)
    return values.tolist()


def main():
    ap = argparse.ArgumentParser()
    for k in ["models", "out", "cache"]:
        ap.add_argument("--" + k, type=Path, required=True)
    ap.add_argument("--teacher", choices=["justrl", "polaris"], required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    train = core.read(a.out / "reference-student.json")
    diagnostic = core.read(a.out / "diagnostic-bank.json")
    test = [r for r in core.read(a.out / "test-bank.json") if r["domain"] == "math"]
    assert len(train) == 128 and len(diagnostic) == 64 and len(test) == 128
    p = core.load(a, "student", student=True)
    q = core.load(a, a.teacher)
    wq = q.lm_head.weight.float()
    fc.cache_hidden(a.cache, a.teacher, q, train + diagnostic + test)
    q.cpu()
    torch.cuda.empty_cache()
    hq = fc.load_hidden(a.cache, a.teacher, train + diagnostic + test, model=q)
    gh, lh = gradient(p, wq, hq, diagnostic)
    nh = products(gh, [])[0] ** 0.5
    gt, lt = gradient(p, wq, hq, test)
    nt = products(gt, [])[0] ** 0.5
    result = []
    for i, r in enumerate(train, 1):
        g, loss = gradient(p, wq, hq, [r])
        norm2, dh, dt = products(g, [gh, gt])
        norm = norm2**0.5
        result.append(
            dict(
                id=r["id"],
                batch=r["id"] // 16 + 1,
                initial_js=loss,
                gradient_norm=norm,
                diagnostic_cosine=dh / (norm * nh),
                test_cosine=dt / (norm * nt),
            )
        )
        del g
        if i % 16 == 0:
            core.dump(
                a.out / f"transfer-proxy-{a.teacher}.json", dict(teacher=a.teacher, records=result, complete=False)
            )
            print("PROXY", a.teacher, i, "/ 128", flush=True)
    for batch in range(1, 9):
        group = sorted([r for r in result if r["batch"] == batch], key=lambda r: r["diagnostic_cosine"])
        for rank, r in enumerate(group):
            r["transfer_bin"] = "low" if rank < 8 else "high"
    core.dump(
        a.out / f"transfer-proxy-{a.teacher}.json",
        dict(
            teacher=a.teacher,
            records=result,
            complete=True,
            parameter_count=len(gh),
            diagnostic_prompts=64,
            test_prompts=128,
            diagnostic_js=lh,
            test_js=lt,
            diagnostic_test_cosine=products(gh, [gt])[1] / (nh * nt),
            definition=(
                "Initial full-parameter JS gradients. Bins use diagnostic cosine only, "
                "ranked within each 16-prompt training batch. Test gradient is "
                "read-only validation of the proxy, not a selection criterion."
            ),
        ),
    )


if __name__ == "__main__":
    main()
