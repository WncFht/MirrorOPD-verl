#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU fixed-prefix diagnostic: initial student, JustRL-1.5B and Polaris.

Uses six deterministic, distinct initial-student validation responses. Sparse
positions across actual prefixes, including the final response token when it
fits, are probes rather than an unbiased token-average or a task evaluation.
Backbones run BF16; selected head projections and softmax run FP32. GPU unused.
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

from examples.mirroropd.assets.paths import registered_model

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--models", type=Path, required=True)
    a.add_argument("--responses", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--terminal-only", action="store_true")
    a.add_argument("--logits-cache", type=Path)
    args = a.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    names = ["DeepSeek-R1-Distill-Qwen-1.5B", "JustRL-DeepSeek-1.5B", "Polaris-7B-Preview"]
    toks = [AutoTokenizer.from_pretrained(registered_model(n, root=args.models), local_files_only=True) for n in names]
    tok = toks[0]
    compat = {
        n: dict(
            vocab_equal=t.get_vocab() == tok.get_vocab(),
            template_equal=t.chat_template == tok.chat_template,
            eos_token_id=t.eos_token_id,
            vocab_len=len(t),
        )
        for n, t in zip(names, toks, strict=False)
    }
    cases = []
    seen = set()
    # Select two short, two medium and two long responses, in source order.
    bins = {0: 0, 1: 0, 2: 0}
    for line in args.responses.open():
        r = json.loads(line)
        if r["input"] in seen:
            continue
        prompt = tok.encode(r["input"], add_special_tokens=False)
        response = tok.encode(r["output"], add_special_tokens=False)
        b = 0 if len(response) < 2048 else (1 if len(response) < 7168 else 2)
        if bins[b] >= 2:
            continue
        seen.add(r["input"])
        bins[b] += 1
        n = min(len(response), 7168)
        # Each start is a next-token action index; score eight adjacent states.
        positions = sorted(
            {j for start in [0, 32, 128, 512, 2048, 4096, 7000, n - 8] for j in range(start, start + 8) if 0 <= j < n}
        )
        cases.append(
            dict(
                id=len(cases),
                uid=r.get("uid"),
                input=r["input"],
                output=r["output"],
                prompt_ids=prompt,
                response_ids=response[:n],
                positions=positions,
                response_length=len(response),
                length_bin=b,
                score=r.get("score"),
            )
        )
        if len(cases) == 6:
            break
    assert len(cases) == 6
    if args.terminal_only:
        cases = [dict(c, positions=[c["response_length"]]) for c in cases if c["response_length"] <= 7168]
    (args.out / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    (args.out / "protocol.json").write_text(
        json.dumps(
            dict(
                source=str(args.responses.resolve()),
                models=names,
                device="cpu",
                threads=8,
                backbone_dtype="bfloat16",
                head_dtype="float32",
                compatibility=compat,
                selection=(
                    "first two distinct prompts in each length bin (<2048,2048..7167,>=7168); selected position blocks"
                ),
                max_response_prefix=7168,
                terminal_only=args.terminal_only,
                torch=torch.__version__,
            ),
            indent=2,
        )
    )
    cached = []
    for name in names:
        start = time.time()
        print("loading", name, flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            registered_model(name, root=args.models),
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        ).eval()
        records = []
        with torch.inference_mode():
            for c in cases:
                ids = torch.tensor([c["prompt_ids"] + c["response_ids"]])
                h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
                offsets = torch.tensor([len(c["prompt_ids"]) - 1 + j for j in c["positions"]])
                # Casting the head makes this a FP32 GEMM on BF16-rounded inputs.
                logits = torch.nn.functional.linear(h[offsets].float(), model.lm_head.weight.float())
                records.append(logits)
                print(
                    name, "case", c["id"], "tokens", ids.numel(), "seconds", round(time.time() - start, 1), flush=True
                )
        if args.logits_cache:
            args.logits_cache.mkdir(parents=True, exist_ok=True)
            torch.save(records, args.logits_cache / (name + ".pt"))
        cached.append(records)
        del model, h, logits
        gc.collect()
    output = []
    for i, c in enumerate(cases):
        lp = cached[0][i].log_softmax(-1)
        p = lp.exp()
        v = lp.shape[-1]
        for teacher, ls in zip(names[1:], cached[1:], strict=False):
            lt = ls[i].log_softmax(-1)
            ltv = lt[:, :v]
            q = lt.exp()
            kl = (p * (lp - ltv)).sum(-1)
            entropy = -(p * lp).sum(-1)
            mass = q[:, v:].sum(-1)
            floor = -torch.log1p(-mass)
            grad = p * (lp - ltv - kl[:, None])
            top = p.topk(32, -1).indices
            tailmask = torch.ones_like(p, dtype=torch.bool).scatter_(1, top, False)
            for j, pos in enumerate(c["positions"]):
                nextid = c["response_ids"][pos] if pos < len(c["response_ids"]) else tok.eos_token_id
                output.append(
                    dict(
                        case=i,
                        position=pos,
                        length_bin=c["length_bin"],
                        teacher=teacher,
                        kl=float(kl[j]),
                        student_entropy=float(entropy[j]),
                        teacher_entropy=float(-(q[j] * lt[j]).sum()),
                        extra_mass=float(mass[j]),
                        support_floor=float(floor[j]),
                        student_next_logp=float(lp[j, nextid]),
                        teacher_next_logp=float(lt[j, nextid]),
                        top1_agree=bool(p[j].argmax() == q[j].argmax()),
                        student_top1=int(p[j].argmax()),
                        teacher_top1=int(q[j].argmax()),
                        student_top1_text=tok.decode([int(p[j].argmax())]),
                        teacher_top1_text=tok.decode([int(q[j].argmax())]),
                        student_stop=float(p[j, 151643] + p[j, 151645]),
                        teacher_stop=float(q[j, 151643] + q[j, 151645]),
                        grad_norm=float(grad[j].norm()),
                        tail_grad_abs_fraction=float(grad[j][tailmask[j]].abs().sum() / grad[j].abs().sum()),
                        tail_student_mass=float(p[j][tailmask[j]].sum()),
                    )
                )
    (args.out / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print("done", len(output), flush=True)


if __name__ == "__main__":
    main()
