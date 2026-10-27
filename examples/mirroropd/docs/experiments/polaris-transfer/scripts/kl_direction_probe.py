#!/usr/bin/env python3
"""Compare KL directions on exactly the same frozen states and initialization.

Teacher is explicitly conditioned on student output support. Full native FKL
would be infinite when teacher has positive mass on rows absent in the student.
Production training code and original model files are not modified.
"""

import argparse
import math
import time
from pathlib import Path
from types import SimpleNamespace

import state_transfer_probe as base
import torch
from transformers import AutoModelForCausalLM


def distributions(sl, tl):
    lp = sl.log_softmax(-1)
    lq = tl[:, : sl.shape[-1]].log_softmax(-1)
    return lp, lq


def distances(sl, tl):
    lp, lq = distributions(sl, tl)
    p, q = lp.exp(), lq.exp()
    r = (p * (lp - lq)).sum(-1)
    f = (q * (lq - lp)).sum(-1)
    lm = torch.logaddexp(lp, lq) - math.log(2)
    return dict(
        reverse=r,
        forward=f,
        js=0.5 * ((p * (lp - lm)).sum(-1) + (q * (lq - lm)).sum(-1)),
        tv=0.5 * (p - q).abs().sum(-1),
        entropy=-(p * lp).sum(-1),
    )


def diagnostic(a):
    out = []
    old = SimpleNamespace(out=a.data, cache=a.cache)
    for r in base.corpus(old):
        sl = base.cached(old, "student", r)
        for teacher in ["justrl", "polaris"]:
            tl = base.cached(old, teacher, r)
            lp, lq = distributions(sl, tl)
            p, q = lp.exp(), lq.exp()
            ds = distances(sl, tl)
            gr = p * (lp - lq - ds["reverse"][:, None])
            gf = p - q
            cosine = (gr * gf).sum(-1) / (gr.norm(dim=-1) * gf.norm(dim=-1)).clamp_min(1e-30)
            lqnative = tl.log_softmax(-1)
            extra = lqnative[:, sl.shape[-1] :].exp().sum(-1)
            fields = {
                **ds,
                "gradient_cosine": cosine,
                "reverse_grad_norm": gr.norm(dim=-1),
                "forward_grad_norm": gf.norm(dim=-1),
                "extra_teacher_mass": extra,
            }
            for k in [16, 64]:
                mask = torch.ones_like(p, dtype=torch.bool).scatter_(1, p.topk(k, -1).indices, False)
                fields[f"teacher_mass_outside_student_top{k}"] = (q * mask).sum(-1)
            for eps in [1e-4, 1e-3]:
                mask = (p < eps) & (q > 0.01)
                fields[f"teacher_mass_p_lt_{eps:g}_q_gt_0.01"] = (q * mask).sum(-1)
                fields[f"forward_abs_gradient_fraction_p_lt_{eps:g}"] = (gf.abs() * mask).sum(-1) / gf.abs().sum(
                    -1
                ).clamp_min(1e-30)
                fields[f"reverse_abs_gradient_fraction_p_lt_{eps:g}"] = (gr.abs() * mask).sum(-1) / gr.abs().sum(
                    -1
                ).clamp_min(1e-30)
            for j, pos in enumerate(r["positions"]):
                out.append(
                    dict(
                        prompt_id=r["id"],
                        source=r["source"],
                        domain=r["domain"],
                        region="early" if pos < 256 else "late",
                        position=pos,
                        teacher=teacher,
                        **{k: float(v[j]) for k, v in fields.items()},
                    )
                )
    base.write(a.out / "gradient-diagnostics.json", out)
    # Verify analytic logit gradients against independent float64 autograd.
    torch.manual_seed(42)
    z = torch.randn(5, 13, dtype=torch.float64, requires_grad=True)
    t = torch.randn(5, 13, dtype=torch.float64)
    lp, lq = distributions(z, t)
    p, q = lp.exp(), lq.exp()
    r = (p * (lp - lq)).sum(-1)
    gr = torch.autograd.grad(r.sum(), z, retain_graph=True)[0]
    gf = torch.autograd.grad((q * (lq - lp)).sum(), z)[0]
    base.write(
        a.out / "gradient-check.json",
        dict(
            reverse_error=float((gr - p * (lp - lq - r[:, None])).abs().max()),
            forward_error=float((gf - (p - q)).abs().max()),
        ),
    )


@torch.no_grad()
def evaluate(a, model, records, step):
    out = []
    old = SimpleNamespace(cache=a.cache)
    for r in records:
        # Fixed full reference context for every before/after comparison.
        sl = base.logits_at(model, r, "cuda:0")
        tl = base.cached(old, a.teacher, r).to("cuda:0")
        d = distances(sl, tl)
        for j, pos in enumerate(r["positions"]):
            out.append(
                dict(
                    step=step,
                    prompt_id=r["id"],
                    source=r["source"],
                    domain=r["domain"],
                    position=pos,
                    region="early" if pos < 256 else "late",
                    **{k: float(v[j]) for k, v in d.items()},
                )
            )
    return out


def train(a):
    torch.cuda.set_per_process_memory_fraction(42 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    while torch.cuda.mem_get_info()[0] < 48 * 2**30:
        print("waiting for 48GiB physical headroom", flush=True)
        time.sleep(10)
    initial = a.checkpoint or a.models / "DeepSeek-R1-Distill-Qwen-1.5B"
    model = (
        AutoModelForCausalLM.from_pretrained(
            initial, dtype=torch.float32, attn_implementation="flash_attention_2", local_files_only=True
        )
        .cuda()
        .eval()
    )
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=1e-6, betas=(0.9, 0.999), weight_decay=0.01, foreach=True)
    records = base.corpus(SimpleNamespace(out=a.data))
    ids = list(range(4 * a.fold, 4 * a.fold + 4))
    training = [r for r in records if r["source"] == "student" and r["id"] in ids]
    arm = f"{a.tag}-{a.teacher}-{a.direction}-fold{a.fold}"
    dest = a.out / arm
    dest.mkdir(exist_ok=True)
    base.write(
        dest / "protocol.json",
        dict(
            initial=str(initial),
            teacher=a.teacher,
            direction=a.direction,
            train_ids=ids,
            steps=a.steps,
            lr=1e-6,
            clip=1,
            optimizer="fresh GPU AdamW FP32",
            target="teacher conditioned on student output support",
            forward_context="fixed complete frozen reference for train and eval; loss only early selected states",
        ),
    )
    ev = evaluate(a, model, records, 0)
    base.write(dest / "evaluation.json", ev)
    history = []
    start = time.time()
    for step in range(1, a.steps + 1):
        model.zero_grad(set_to_none=True)
        value = 0.0
        total = sum(sum(p < 256 for p in r["positions"]) for r in training)
        for r in training:
            ix = [i for i, p in enumerate(r["positions"]) if p < 256]
            pos = [r["positions"][i] for i in ix]
            sl = base.logits_at(model, r, "cuda:0", pos, train=True, context_limit=max(r["positions"]))
            tl = base.cached(SimpleNamespace(cache=a.cache), a.teacher, r)[ix].cuda()
            lp, lq = distributions(sl, tl)
            loss = ((lp.exp() * (lp - lq)).sum() if a.direction == "reverse" else (lq.exp() * (lq - lp)).sum()) / total
            loss.backward()
            value += float(loss.detach())
            del sl, tl, loss, lp, lq
        grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
        assert math.isfinite(value) and torch.isfinite(grad)
        before = [p.detach().cpu() for p in params] if step in [1, 8, a.steps] else None
        opt.step()
        model.zero_grad(set_to_none=True)
        row = dict(
            step=step,
            loss_before=value,
            grad_norm=float(grad),
            seconds=time.time() - start,
            peak_gib=torch.cuda.max_memory_allocated() / 2**30,
        )
        if before is not None:
            delta2 = sum(float((p.detach().cpu() - old).square().sum()) for p, old in zip(params, before))
            row["parameter_step_norm"] = delta2**0.5
            del before
        history.append(row)
        base.write(dest / "training.json", history)
        print(arm, row, flush=True)
        if step in [1, 8, a.steps]:
            ev.extend(evaluate(a, model, records, step))
            base.write(dest / "evaluation.json", ev)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["diagnostic", "train"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--teacher", choices=["justrl", "polaris"], default="polaris")
    p.add_argument("--direction", choices=["forward", "reverse"], default="forward")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--tag", default="initial")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    globals()[a.stage](a)


if __name__ == "__main__":
    main()
