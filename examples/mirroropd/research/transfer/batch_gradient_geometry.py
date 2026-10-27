#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact full-parameter gradient Gram across four disjoint prompt batches.

Each batch is eight prompts times two rollouts. Gradients are stored on CPU;
Gram products are streamed in FP32 chunks and accumulated in FP64. No sketches
or head-only proxies. Raw gradients are not Adam parameter-update directions.
"""

import argparse
import gc
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from ..common import native as core


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--previous-cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--warm-step", type=int, default=128)
    p.add_argument("--warm-template", default="{teacher}-blocked")
    p.add_argument("--prompt-start", type=int, default=64)
    p.add_argument("--panel", choices=["fresh", "fixed", "coverage"], default="fresh")
    p.add_argument("--case-index", type=int)
    p.add_argument("--new-limit", type=int, default=448)
    a = p.parse_args()
    a.cache = a.previous_cache
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.85)
    output = []
    cases = [
        ("initial", "justrl"),
        ("initial", "polaris"),
        (f"own{a.warm_step}", "justrl"),
        (f"own{a.warm_step}", "polaris"),
        (f"polaris{a.warm_step}", "justrl"),
    ]
    for case_index, (stage, target) in enumerate(cases):
        if a.case_index is not None and case_index != a.case_index:
            continue
        a.teacher = target
        if stage == "initial":
            initial = a.models / core.NAMES["student"]
            records = core.read(a.previous / f"baseline-{target}/fresh.json")
        else:
            origin = target if stage.startswith("own") else "polaris"
            arm = a.warm_template.format(teacher=origin)
            initial = a.previous_cache / "checkpoints" / arm
            records = core.read(a.previous / arm / f"fresh-{a.warm_step}.json")
        if a.panel in ["fixed", "coverage"]:
            records = core.read(a.previous / "bank.json")
        if a.panel == "coverage":
            records += sum([core.read(a.previous / f"rich-shard-{i}.json") for i in range(4)], [])
        inputs = a.out / "geometry-inputs"
        inputs.mkdir(exist_ok=True)
        core.dump(inputs / f"{stage}-{target}-{a.panel}.json", records)
        model = (
            AutoModelForCausalLM.from_pretrained(
                initial, dtype=torch.float32, attn_implementation="flash_attention_2", local_files_only=True
            )
            .cuda()
            .eval()
        )
        teacher = core.load(a, target)
        wq = teacher.lm_head.weight.float()
        grads = []
        losses = []
        started = time.time()
        if a.panel == "coverage":
            groups = [
                ("Seen64", [r for r in records if r["id"] < 64]),
                (f"New{a.new_limit}", [r for r in records if r["id"] >= 160][: a.new_limit]),
                ("Held64", [r for r in records if r["split"] == "heldout" and r["domain"] == "math"]),
            ]
        elif a.panel == "fixed":
            groups = [
                ("TrainMath", [r for r in records if r["split"] == "train"]),
                ("HeldMath", [r for r in records if r["split"] == "heldout" and r["domain"] == "math"]),
                ("Code", [r for r in records if r["domain"] == "code"]),
                ("IF", [r for r in records if r["domain"] == "if"]),
            ]
        else:
            groups = [
                (f"G{i + 1}", [r for r in records if a.prompt_start + 8 * i <= r["id"] < a.prompt_start + 8 * (i + 1)])
                for i in range(4)
            ]
        for group, (group_name, rows) in enumerate(groups):
            assert len(rows) == (
                [64, a.new_limit, 64][group]
                if a.panel == "coverage"
                else ([64, 64, 16, 16][group] if a.panel == "fixed" else 16)
            )
            model.zero_grad(set_to_none=True)
            value = 0.0
            for r in rows:
                with torch.no_grad():
                    hq = (
                        core.teacher_hidden(a, teacher, r)
                        if a.panel in ["fixed", "coverage"]
                        else core.hidden(teacher, r["prompt_ids"] + r["response_ids"][:-1])[len(r["prompt_ids"]) - 1 :]
                    )
                hp = core.hidden(model, r["prompt_ids"] + r["response_ids"][:-1])[len(r["prompt_ids"]) - 1 :]
                terms = []
                for start in range(0, len(hp), 128):
                    sp = torch.nn.functional.linear(hp[start : start + 128].float(), model.lm_head.weight)
                    with torch.no_grad():
                        lq = torch.nn.functional.linear(hq[start : start + 128].float(), wq).log_softmax(-1)[
                            :, : sp.shape[-1]
                        ]
                    lp = sp.log_softmax(-1)
                    terms.append((lp.exp() * (lp - lq)).sum() / (len(rows) * 1024))
                loss = sum(terms)
                loss.backward()
                value += float(loss.detach())
                del hp, hq, sp, lp, lq, loss, terms
            gradient = torch.cat(
                [
                    (v.grad.detach() if v.grad is not None else torch.zeros_like(v)).reshape(-1)
                    for v in model.parameters()
                ]
            ).cpu()
            grads.append(gradient)
            losses.append(value)
            model.zero_grad(set_to_none=True)
            print("gradient", stage, target, group, "loss", value, "seconds", time.time() - started, flush=True)
        gram = torch.zeros((len(groups), len(groups)), device="cuda", dtype=torch.float64)
        for start in range(0, len(grads[0]), 2**20):
            chunk = torch.stack([v[start : start + 2**20] for v in grads]).cuda()
            gram += (chunk @ chunk.T).double()
            del chunk
        g = gram.cpu()
        norm = g.diag().sqrt()
        cos = g / (norm[:, None] * norm[None, :])
        n = len(groups)
        output.append(
            dict(
                stage=stage,
                target=target,
                checkpoint=str(initial),
                panel=a.panel,
                groups=[name for name, _ in groups],
                group_sizes=[len(rows) for _, rows in groups],
                losses=losses,
                gradient_norms=norm.tolist(),
                gram=g.tolist(),
                cosine=cos.tolist(),
                mean_offdiagonal_cosine=float((cos.sum() - cos.diag().sum()) / (n * (n - 1))),
                coherence=float(g.sum() / (n * g.diag().sum())),
                parameter_count=len(grads[0]),
                seconds=time.time() - started,
            )
        )
        filename = (
            ("coverage-gradient.json" if a.new_limit == 448 else f"coverage-gradient-{a.new_limit}.json")
            if a.panel == "coverage"
            else (f"fixed-gradient-case-{case_index}.json" if a.panel == "fixed" else "batch-gradient-geometry.json")
        )
        core.dump(a.out / filename, output if a.panel == "fresh" else [output[-1]])
        del model, teacher, wq, grads, gradient, gram
        gc.collect()
        torch.cuda.empty_cache()
    if a.panel == "fresh":
        core.dump(a.out / "batch-gradient-geometry-complete.json", dict(cases=len(output), finished=True))


if __name__ == "__main__":
    main()
