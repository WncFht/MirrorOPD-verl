#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Small crossed rollout-source / target-teacher experiment.

No serving engine or production training changes. Frozen probes have independent
prompt IDs, early and later states, and math/code/IF domains. Full-parameter
student updates use FP32 master weights and CPU AdamW states, BF16 backbone
autocast, and FP32 output projection. GPU allocator capped at 20 GiB.
"""

import argparse
import gc
import json
import random
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..common.io import dump as write

MODELS = {"student": "DeepSeek-R1-Distill-Qwen-1.5B", "justrl": "JustRL-DeepSeek-1.5B", "polaris": "Polaris-7B-Preview"}


def load_model(args, which, train=False):
    # Other jobs share these devices. Check physical headroom before loading;
    # allocator limits alone do not reserve memory for those jobs.
    required = (18 if args.stage == "geometry" else (24 if train else (18 if which == "polaris" else 6))) * 2**30
    while torch.cuda.mem_get_info(args.device)[0] < required:
        print("waiting for GPU headroom", which, round(torch.cuda.mem_get_info(args.device)[0] / 2**30, 2), flush=True)
        time.sleep(10)
    model = AutoModelForCausalLM.from_pretrained(
        args.models / MODELS[which],
        dtype=torch.float32 if train else torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    ).to(args.device)
    return model.eval()


def prepare(args):
    tok = AutoTokenizer.from_pretrained(args.models / MODELS["student"], local_files_only=True)
    records = []
    rng = random.Random(20261027)
    for domain, path, count in [
        ("math", "math/DAPO/train.parquet", 16),
        ("code", "code/TACO/train.parquet", 4),
        ("if", "if/MultiIF/train.parquet", 4),
    ]:
        rows = pq.read_table(args.data / path, columns=["prompt", "extra_info"]).to_pylist()
        rng.shuffle(rows)
        selected = 0
        for row in rows:
            prompt = tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
            ids = tok.encode(prompt, add_special_tokens=False)
            if not 32 <= len(ids) <= 384:
                continue
            if (
                any("\u4e00" <= c <= "\u9fff" for c in prompt)
                or "http" in prompt
                or "interactive problem" in prompt.lower()
            ):
                continue
            if domain == "if" and "English" not in str(row["extra_info"]):
                continue
            records.append(
                dict(
                    id=len(records),
                    domain=domain,
                    split=("train" if selected < 8 else "heldout") if domain == "math" else "heldout",
                    prompt=prompt,
                    prompt_ids=ids,
                    source_index=row["extra_info"].get("index"),
                )
            )
            selected += 1
            if selected == count:
                break
        assert selected == count
    write(args.out / "prompts.json", records)
    write(
        args.out / "protocol.json",
        dict(
            seed=20261027,
            models=MODELS,
            max_new_tokens=args.cap,
            temperature=1.0,
            top_p=1.0,
            max_prompt_tokens=384,
            domains={"math": 16, "code": 4, "if": 4},
            train_prompt_count=8,
            generation_dtype="bfloat16",
            attention="flash_attention_2",
            early_positions="0:256:8",
            late_positions="512:768:16",
            note="Short fixed-prefix mechanism probe; no final task-accuracy claim or long-horizon reproduction.",
        ),
    )


def generate(args):
    prompts = json.loads((args.out / "prompts.json").read_text())
    final = args.out / f"rollouts-{args.source}.json"
    if final.exists() and len(json.loads(final.read_text())) == len(prompts):
        print("complete rollout cache exists", args.source, flush=True)
        return
    tok = AutoTokenizer.from_pretrained(args.models / MODELS[args.source], local_files_only=True)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    model = load_model(args, args.source)
    records = []
    started = time.time()
    for start in range(0, len(prompts), 8):
        batch = prompts[start : start + 8]
        torch.manual_seed(20261027 + start)
        enc = tok([p["prompt"] for p in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(
            args.device
        )
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=args.cap,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                use_cache=True,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
        for p, t in zip(batch, out[:, enc.input_ids.shape[1] :].tolist(), strict=False):
            eos = tok.eos_token_id in t
            if eos:
                t = t[: t.index(tok.eos_token_id) + 1]
            positions = list(range(0, min(256, len(t)), 8)) + list(range(512, min(768, len(t)), 16))
            records.append(
                dict(
                    p,
                    source=args.source,
                    response_ids=t,
                    output=tok.decode(t, skip_special_tokens=False),
                    positions=positions,
                    finish_reason="stop" if eos else "length",
                )
            )
        print("generated", args.source, len(records), "seconds", round(time.time() - started, 2), flush=True)
        write(args.out / f"rollouts-{args.source}.json", records)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def corpus(args):
    return [r for src in MODELS for r in json.loads((args.out / f"rollouts-{src}.json").read_text())]


def logits_at(model, r, device, positions=None, train=False, context_limit=None):
    pos = r["positions"] if positions is None else positions
    # A state at action index p only needs response tokens strictly before p.
    tokens = r["prompt_ids"] + r["response_ids"][: max(pos) if context_limit is None else context_limit]
    ids = torch.tensor([tokens], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
    take = torch.tensor([len(r["prompt_ids"]) - 1 + p for p in pos], device=device)
    # FP32 accumulation on BF16-rounded hidden/head, same precision intent as run.
    return torch.nn.functional.linear(h[take].float(), model.lm_head.weight.to(torch.bfloat16).float())


def score(args):
    records = corpus(args)
    model = load_model(args, args.source)
    started = time.time()
    cachedir = args.cache / args.source
    cachedir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for i, r in enumerate(records):
            target = cachedir / f"{r['source']}-{r['id']:02d}.pt"
            if target.exists():
                continue
            tokens = r["prompt_ids"] + r["response_ids"][:-1]
            h = model.model(input_ids=torch.tensor([tokens], device=args.device), use_cache=False).last_hidden_state[0]
            h = h[len(r["prompt_ids"]) - 1 :]
            head = model.lm_head.weight.float()
            chosen = {}
            sample_logp = []
            entropies = []
            for start in range(0, len(h), 128):
                logits = torch.nn.functional.linear(h[start : start + 128].float(), head)
                lp = logits.log_softmax(-1)
                nextids = torch.tensor(r["response_ids"][start : start + len(lp)], device=args.device)
                sample_logp.extend(lp.gather(1, nextids[:, None]).squeeze(1).tolist())
                entropies.extend((-(lp.exp() * lp).sum(-1)).tolist())
                for pos in r["positions"]:
                    if start <= pos < start + len(lp):
                        chosen[pos] = logits[pos - start].cpu()
            torch.save(torch.stack([chosen[p] for p in r["positions"]]), target)
            write(target.with_suffix(".json"), dict(sample_logp=sample_logp, entropy=entropies))
            del h, head, chosen, lp, logits
            if i % 8 == 0:
                print("scored", args.source, i, "seconds", round(time.time() - started, 1), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def cached(args, teacher, r):
    return torch.load(args.cache / teacher / f"{r['source']}-{r['id']:02d}.pt", weights_only=True)


def kl_rows(sl, tl, temperature=1.0):
    lp = sl.log_softmax(-1)
    lt = (tl / temperature).log_softmax(-1)[:, : sl.shape[-1]]
    return (lp.exp() * (lp - lt)).sum(-1)


def metrics(sl, tl, ids):
    lp = sl.log_softmax(-1)
    p = lp.exp()
    lt = tl.log_softmax(-1)
    q = lt.exp()
    v = sl.shape[-1]
    kl = (p * (lp - lt[:, :v])).sum(-1)
    grad = p * (lp - lt[:, :v] - kl[:, None])
    data = {
        "kl": kl,
        "entropy_student": -(p * lp).sum(-1),
        "entropy_teacher": -(q * lt).sum(-1),
        "student_max_probability": p.max(-1).values,
        "teacher_max_probability": q.max(-1).values,
        "total_variation": 0.5 * ((p - q[:, :v]).abs().sum(-1) + q[:, v:].sum(-1)),
        "sample_log_gap": lp.gather(1, ids[:, None]).squeeze(1) - lt.gather(1, ids[:, None]).squeeze(1),
        "positive_sample_signal": lt.gather(1, ids[:, None]).squeeze(1) > lp.gather(1, ids[:, None]).squeeze(1),
        "logit_gradient_norm": grad.norm(dim=-1),
        "extra_teacher_mass": q[:, v:].sum(-1),
        "positive_signal_mass_under_student": (p * (q[:, :v] > p)).sum(-1),
    }
    for k in [1, 8, 16, 32, 64]:
        si = p.topk(k, -1).indices
        ti = q.topk(k, -1).indices
        common = (si[:, :, None] == ti[:, None, :]).any(-1)
        massp = (p.gather(1, si) * common).sum(-1)
        massq = (q.gather(1, si) * common).sum(-1)
        data[f"overlap{k}"] = common.float().mean(-1)
        data[f"overlap_student_mass{k}"] = massp
        data[f"overlap_teacher_mass{k}"] = massq
        if k == 16:
            pc = p.gather(1, si) * common
            qc = q.gather(1, si) * common
            pn = pc / massp.clamp_min(1e-30)[:, None]
            qn = qc / massq.clamp_min(1e-30)[:, None]
            conditional = (pn * (pn.clamp_min(1e-30).log() - qn.clamp_min(1e-30).log())).sum(-1)
            data["overlap_conditional_kl16"] = conditional
            data["overlap_advantage16"] = -conditional / common.sum(-1).clamp_min(1)
            data["overlap_empty16"] = common.sum(-1) == 0
    return {k: v.tolist() for k, v in data.items()}


def diagnose(args):
    output = []
    for r in corpus(args):
        sl = cached(args, "student", r)
        for teacher in ["justrl", "polaris"]:
            tl = cached(args, teacher, r)
            ids = torch.tensor([r["response_ids"][p] for p in r["positions"]])
            d = metrics(sl, tl, ids)
            for i, pos in enumerate(r["positions"]):
                output.append(
                    dict(
                        prompt_id=r["id"],
                        domain=r["domain"],
                        split=r["split"],
                        source=r["source"],
                        teacher=teacher,
                        position=pos,
                        region="early" if pos < 256 else "late",
                        **{k: v[i] for k, v in d.items()},
                    )
                )
    write(args.out / "initial-state-metrics.json", output)


@torch.no_grad()
def evaluate(args, model, records, step):
    rows = []
    for r in records:
        sl = logits_at(model, r, args.device)
        tl = cached(args, args.teacher, r).to(args.device)
        d = metrics(sl, tl, torch.tensor([r["response_ids"][p] for p in r["positions"]], device=args.device))
        for i, pos in enumerate(r["positions"]):
            rows.append(
                dict(
                    step=step,
                    prompt_id=r["id"],
                    domain=r["domain"],
                    split=r["split"],
                    source=r["source"],
                    position=pos,
                    region="early" if pos < 256 else "late",
                    **{k: v[i] for k, v in d.items()},
                )
            )
        del sl, tl
    return rows


def train(args):
    records = corpus(args)
    selected = [
        r for r in records if r["source"] == args.source and r["domain"] == "math" and r["id"] in args.train_ids
    ]
    assert len(selected) == len(args.train_ids)
    arm = f"{args.teacher}-{args.source}-split{args.split}-t{args.temperature:g}-lr{args.lr:g}"
    dest = args.out / arm
    dest.mkdir(exist_ok=True)
    model = load_model(args, "student", train=True)
    gpuparams = [p for p in model.parameters() if p.requires_grad]
    cpuparams = [torch.nn.Parameter(p.detach().cpu().clone()) for p in gpuparams]
    opt = torch.optim.AdamW(cpuparams, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01, foreach=True)
    write(
        dest / "protocol.json",
        dict(
            teacher=args.teacher,
            source=args.source,
            train_ids=args.train_ids,
            steps=args.steps,
            lr=args.lr,
            teacher_temperature=args.temperature,
            optimizer="CPU AdamW exact FP32 states",
            params=sum(p.numel() for p in gpuparams),
            frozen_rollouts=True,
        ),
    )
    history = []
    start = time.time()
    baseline = args.out / f"fp32-master-baseline-{args.teacher}.json"
    if baseline.exists():
        evalrows = json.loads(baseline.read_text())
    else:
        evalrows = evaluate(args, model, records, 0)
        write(baseline, evalrows)
    write(dest / "evaluation.json", evalrows)
    for step in range(1, args.steps + 1):
        model.zero_grad(set_to_none=True)
        lossvalue = 0.0
        individual_norms = []
        previous = None
        total = sum(sum(p < 256 for p in r["positions"]) for r in selected)
        for r in selected:
            indices = [i for i, p in enumerate(r["positions"]) if p < 256]
            positions = [r["positions"][i] for i in indices]
            sl = logits_at(model, r, args.device, positions, train=True)
            tl = cached(args, args.teacher, r)[indices].to(args.device)
            loss = kl_rows(sl, tl, args.temperature).sum() / total
            loss.backward()
            lossvalue += float(loss.detach())
            del sl, tl, loss
            if step == 1:
                current = [p.grad.detach().cpu() if p.grad is not None else None for p in gpuparams]
                normsq = 0.0
                for j, g in enumerate(current):
                    if g is not None:
                        individual = g if previous is None or previous[j] is None else g - previous[j]
                        normsq += float(individual.square().sum())
                individual_norms.append(normsq**0.5)
                previous = current
        norm = torch.nn.utils.clip_grad_norm_(gpuparams, 1.0)
        assert torch.isfinite(norm) and __import__("math").isfinite(lossvalue)
        for gp, cp in zip(gpuparams, cpuparams, strict=False):
            cp.grad = gp.grad.detach().cpu() if gp.grad is not None else None
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for gp, cp in zip(gpuparams, cpuparams, strict=False):
                gp.copy_(cp)
        model.zero_grad(set_to_none=True)
        history.append(
            dict(
                step=step,
                pre_update_train_kl=lossvalue,
                grad_norm=float(norm),
                seconds=time.time() - start,
                allocated_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
            )
        )
        if step == 1:
            sumsq = sum(x * x for x in individual_norms)
            normtotal = float(norm) ** 2
            history[-1].update(
                individual_gradient_norms=individual_norms,
                gradient_coherence=normtotal / (len(selected) * sumsq),
                norm_weighted_pairwise_cosine=(normtotal - sumsq) / (sum(individual_norms) ** 2 - sumsq),
            )
            del previous, current, individual
        print("train", arm, history[-1], flush=True)
        write(dest / "training.json", history)
        if step in [1, 8, args.steps]:
            evalrows.extend(evaluate(args, model, records, step))
            write(dest / "evaluation.json", evalrows)
    # No production model checkpoint is changed or saved.
    del opt, model, cpuparams, gpuparams
    gc.collect()
    torch.cuda.empty_cache()


def geometry(args):
    records = [r for r in corpus(args) if r["source"] == "student" and r["domain"] == "math" and r["id"] < 8]
    model = load_model(args, "student", train=True)
    params = list(model.parameters())
    output = []
    # Capture completed leaf gradients on CPU to leave room for co-tenants.
    # This changes storage only; each full FP32 leaf gradient is retained.
    captured = [None] * len(params)

    def capture(index):
        def hook(parameter):
            captured[index] = parameter.grad.detach().cpu()
            parameter.grad = None

        return hook

    _hooks = [parameter.register_post_accumulate_grad_hook(capture(i)) for i, parameter in enumerate(params)]
    for teacher in ["justrl", "polaris"]:
        for r in records:
            norms = []
            summed = None
            for start in [0, 64, 128, 192]:
                positions = [p for p in r["positions"] if start <= p < start + 64]
                indices = [r["positions"].index(p) for p in positions]
                model.zero_grad(set_to_none=True)
                sl = logits_at(model, r, args.device, positions, train=True, context_limit=248)
                tl = cached(args, teacher, r)[indices].to(args.device)
                loss = kl_rows(sl, tl).mean()
                loss.backward()
                normsq = 0.0
                if summed is None:
                    summed = [torch.zeros_like(p, device="cpu") for p in params]
                for own, g in zip(captured, summed, strict=False):
                    if own is not None:
                        normsq += float(own.square().sum())
                        g.add_(own)
                norms.append(normsq**0.5)
                del sl, tl, loss
            total = sum(float(g.square().sum()) for g in summed)
            ss = sum(v * v for v in norms)
            output.append(
                dict(
                    teacher=teacher,
                    prompt_id=r["id"],
                    units="four consecutive 64-token spans; eight scored states each; fixed response-prefix length 248",
                    gradient_norms=norms,
                    coherence=total / (4 * ss),
                    norm_weighted_pairwise_cosine=(total - ss) / (sum(norms) ** 2 - ss),
                )
            )
            output[-1]["allocated_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
            print("geometry", output[-1], flush=True)
            write(args.out / "within-trace-gradient-geometry.json", output)
            del summed


def main():
    a = argparse.ArgumentParser()
    a.add_argument("stage", choices=["prepare", "generate", "score", "diagnose", "train", "geometry"])
    a.add_argument("--models", type=Path, required=True)
    a.add_argument("--data", type=Path)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--cache", type=Path, required=True)
    a.add_argument("--source", choices=list(MODELS), default="student")
    a.add_argument("--teacher", choices=["justrl", "polaris"], default="polaris")
    a.add_argument("--device", default="cuda:0")
    a.add_argument("--cap", type=int, default=768)
    a.add_argument("--steps", type=int, default=16)
    a.add_argument("--split", type=int, default=0)
    a.add_argument("--lr", type=float, default=1e-6)
    a.add_argument("--temperature", type=float, default=1.0)
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261027)
    if args.stage not in ["prepare", "diagnose"]:
        torch.cuda.set_device(args.device)
        total = torch.cuda.get_device_properties(args.device).total_memory
        torch.cuda.set_per_process_memory_fraction(20 * 2**30 / total, args.device)
    args.train_ids = list(range(args.split * 4, args.split * 4 + 4))
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
