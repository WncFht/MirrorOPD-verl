#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Separate the nominal 7B base mismatch from Polaris post-training changes.

This is a fixed-initial-student logit-gradient decomposition, not an additive
decomposition of KL values or of capability transfer.
"""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.mirroropd.assets.paths import registered_model
from examples.mirroropd.research.core.io import dump
from examples.mirroropd.research.studies.polaris_transfer.kl_direction_probe import distributions
from examples.mirroropd.research.studies.state_sources import state_transfer_probe as base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(24 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    tok = AutoTokenizer.from_pretrained(
        registered_model("DeepSeek-R1-Distill-Qwen-7B", root=a.models), local_files_only=True
    )
    studenttok = AutoTokenizer.from_pretrained(
        registered_model("DeepSeek-R1-Distill-Qwen-1.5B", root=a.models), local_files_only=True
    )
    assert tok.get_vocab() == studenttok.get_vocab() and tok.chat_template == studenttok.chat_template
    model = (
        AutoModelForCausalLM.from_pretrained(
            registered_model("DeepSeek-R1-Distill-Qwen-7B", root=a.models),
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            local_files_only=True,
        )
        .cuda()
        .eval()
    )
    rs = [r for r in base.corpus(SimpleNamespace(out=a.data)) if r["source"] == "student"]
    result = []
    geometry = {}
    with torch.inference_mode():
        for r in rs:
            lb = base.logits_at(model, r, "cuda:0").cpu()
            ls = base.cached(SimpleNamespace(cache=a.cache), "student", r)
            lt = base.cached(SimpleNamespace(cache=a.cache), "polaris", r)
            lp, lqb = distributions(ls, lb)
            _, lqt = distributions(ls, lt)
            prob = lp.exp()
            qb = lqb.exp()
            qt = lqt.exp()
            krb = (prob * (lp - lqb)).sum(-1)
            krt = (prob * (lp - lqt)).sum(-1)
            gb = prob * (lp - lqb - krb[:, None])
            gt = prob * (lp - lqt - krt[:, None])
            gd = gt - gb
            delta = lqt - lqb
            explicit = -prob * (delta - (prob * delta).sum(-1)[:, None])
            for i, pos in enumerate(r["positions"]):
                result.append(
                    dict(
                        prompt_id=r["id"],
                        domain=r["domain"],
                        position=pos,
                        region="early" if pos < 256 else "late",
                        student_to_base_reverse=float(krb[i]),
                        student_to_polaris_reverse=float(krt[i]),
                        polaris_to_base_reverse=float((qt[i] * (lqt[i] - lqb[i])).sum()),
                        base_to_polaris_reverse=float((qb[i] * (lqb[i] - lqt[i])).sum()),
                        base_gradient_norm=float(gb[i].norm()),
                        polaris_gradient_norm=float(gt[i].norm()),
                        delta_gradient_norm=float(gd[i].norm()),
                        base_dot_polaris=float((gb[i] * gt[i]).sum()),
                        delta_dot_polaris=float((gd[i] * gt[i]).sum()),
                        delta_identity_error=float((gd[i] - explicit[i]).abs().max()),
                    )
                )
            print("scored base7", r["id"], flush=True)
    for domain in ["math", "code", "if"]:
        rows = [r for r in result if r["domain"] == domain]
        n = len(rows)
        b2 = sum(r["base_gradient_norm"] ** 2 for r in rows)
        t2 = sum(r["polaris_gradient_norm"] ** 2 for r in rows)
        d2 = sum(r["delta_gradient_norm"] ** 2 for r in rows)
        bt = sum(r["base_dot_polaris"] for r in rows)
        dt = sum(r["delta_dot_polaris"] for r in rows)
        geometry[domain] = dict(
            count=n,
            student_to_base_reverse=sum(r["student_to_base_reverse"] for r in rows) / n,
            student_to_polaris_reverse=sum(r["student_to_polaris_reverse"] for r in rows) / n,
            polaris_to_base_reverse=sum(r["polaris_to_base_reverse"] for r in rows) / n,
            base_gradient_projection_on_total=bt / t2,
            delta_gradient_projection_on_total=dt / t2,
            base_to_total_gradient_cosine=bt / math.sqrt(b2 * t2),
            delta_to_total_gradient_cosine=dt / math.sqrt(d2 * t2),
            delta_identity_max_error=max(r["delta_identity_error"] for r in rows),
        )
    dump(a.out / "lineage-state-metrics.json", result)
    dump(a.out / "lineage-summary.json", geometry)
    print(json.dumps(geometry, indent=2))


if __name__ == "__main__":
    main()
