#!/usr/bin/env python3
"""Online short training with independently frozen evaluation measures.

Sampling, support diagnostics, and loss support are separate: intersection
sampling conditions p on S_k intersect T_k, while every training loss remains
full-vocabulary reverse KL. Fixed panels never depend on the training arm.
"""

import argparse
import gc
import json
import math
import os
import random
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NAMES = {"student": "DeepSeek-R1-Distill-Qwen-1.5B", "justrl": "JustRL-DeepSeek-1.5B", "polaris": "Polaris-7B-Preview"}
KS = (8, 16, 64)


def dump(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read(path):
    return json.loads(path.read_text())


def prepare(a):
    tok = AutoTokenizer.from_pretrained(a.models / NAMES["student"], local_files_only=True)
    rng = random.Random(2026102702)
    rows = []
    for domain, file, count, train in [
        ("math", "math/DAPO/train.parquet", 192, 128),
        ("code", "code/TACO/train.parquet", 16, 0),
        ("if", "if/MultiIF/train.parquet", 16, 0),
    ]:
        raw = pq.read_table(a.data / file, columns=["prompt", "extra_info", "reward_model"]).to_pylist()
        rng.shuffle(raw)
        n = 0
        seen = set()
        for r in raw:
            text = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
            ids = tok.encode(text, add_special_tokens=False)
            if not 24 <= len(ids) <= 384 or text in seen:
                continue
            if any("\u4e00" <= c <= "\u9fff" for c in text) or "http" in text or "interactive problem" in text.lower():
                continue
            if domain == "if" and "English" not in str(r["extra_info"]):
                continue
            seen.add(text)
            rows.append(
                dict(
                    id=len(rows),
                    domain=domain,
                    split="train" if n < train else "heldout",
                    prompt=text,
                    prompt_ids=ids,
                    source_index=r["extra_info"].get("index"),
                    ground_truth=r["reward_model"]["ground_truth"],
                )
            )
            n += 1
            if n == count:
                break
        assert n == count
    dump(a.out / "prompts.json", rows)
    import re

    existing = {re.sub(r"\s+", " ", r["prompt"]).strip() for r in rows}
    task_raw = pq.read_table(a.data / "math/MATH-500/test.parquet").to_pylist()
    random.Random(54127).shuffle(task_raw)
    task = []
    for r in task_raw:
        text = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) > 384 or re.sub(r"\s+", " ", text).strip() in existing:
            continue
        task.append(
            dict(
                id=len(task),
                domain="math500",
                split="task_eval",
                prompt=text,
                prompt_ids=ids,
                source_index=r["extra_info"].get("index"),
                ground_truth=r["reward_model"]["ground_truth"],
            )
        )
        if len(task) == 32:
            break
    dump(a.out / "task-prompts.json", task)
    dump(
        a.out / "protocol.json",
        dict(
            seed=2026102702,
            models=NAMES,
            cap=a.cap,
            training_prompts=128,
            heldout={"math": 64, "code": 16, "if": 16},
            training_batch=32,
            reuse=16,
            outers=4,
            updates=64,
            lr=1e-6,
            betas=[0.9, 0.999],
            weight_decay=0.01,
            clip=1.0,
            loss="full-vocabulary reverse KL, all response positions",
            precision="FP32 student parameters and LM head; BF16 backbone autocast; BF16 frozen teacher with FP32 head projection",
            fixed_reference_sources=list(NAMES),
            fresh_panel="first 32 heldout math prompts, two generation seeds",
            sampling="temperature=1, unrestricted student proposals; k=16 conditional resampling for constrained arms",
            categories=["neither", "teacher_only", "student_only", "both"],
        ),
    )


def load(a, name, student=False):
    m = (
        AutoModelForCausalLM.from_pretrained(
            a.models / NAMES[name],
            dtype=torch.float32 if student else torch.bfloat16,
            attn_implementation="flash_attention_2",
            local_files_only=True,
        )
        .to("cuda")
        .eval()
    )
    if not student:
        m.requires_grad_(False)
    return m


def hidden(m, tokens):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return m.model(input_ids=torch.tensor([tokens], device="cuda"), use_cache=False).last_hidden_state[0]


def vectors(zp, zt, actions):
    lp = zp.float().log_softmax(-1)
    lq = zt.float().log_softmax(-1)
    p = lp.exp()
    q = lq.exp()
    v = p.shape[-1]
    qc = q[:, :v]
    lm = torch.logaddexp(lp, lq[:, :v]) - math.log(2)
    out = {
        "rkl": (p * (lp - lq[:, :v])).sum(-1),
        "js": 0.5 * ((p * (lp - lm)).sum(-1) + (qc * (lq[:, :v] - lm)).sum(-1) + q[:, v:].sum(-1) * math.log(2)),
        "tv": 0.5 * ((p - qc).abs().sum(-1) + q[:, v:].sum(-1)),
        "hp": -(p * lp).sum(-1),
        "hq": -(q * lq).sum(-1),
        "sample_gap": lp.gather(1, actions[:, None]).squeeze(1) - lq.gather(1, actions[:, None]).squeeze(1),
    }
    maxsp = p.topk(max(KS), -1).indices
    maxtq = q.topk(max(KS), -1).indices
    for k in KS:
        sp = maxsp[:, :k]
        tq = maxtq[:, :k]
        common = (sp[:, :, None] == tq[:, None, :]).any(-1)
        inter = common.sum(-1)
        pc = (p.gather(1, sp) * common).sum(-1)
        qt = (q.gather(1, sp) * common).sum(-1)
        ps = p.gather(1, sp).sum(-1)
        qs = q.gather(1, sp).sum(-1)
        pt = p.gather(1, tq.clamp(max=v - 1)).mul(tq < v).sum(-1)
        qq = q.gather(1, tq).sum(-1)
        ins = (sp == actions[:, None]).any(-1)
        int_ = (tq == actions[:, None]).any(-1)
        cat = 2 * ins.long() + int_.long()
        out[f"overlap{k}"] = inter.float() / k
        out[f"jaccard{k}"] = inter.float() / (2 * k - inter)
        out[f"common_p{k}"] = pc
        out[f"common_q{k}"] = qt
        for c, pm, qm in [
            (0, 1 - ps - pt + pc, 1 - qs - qq + qt),
            (1, pt - pc, qq - qt),
            (2, ps - pc, qs - qt),
            (3, pc, qt),
        ]:
            out[f"category{k}_{c}"] = (cat == c).float()
            out[f"mass_p{k}_{c}"] = pm
            out[f"mass_q{k}_{c}"] = qm
    return out


@torch.no_grad()
def decode(a, student, teacher, prompts, mode, seed, cap=None):
    cap = a.cap if cap is None else cap
    tok = AutoTokenizer.from_pretrained(a.models / NAMES["student"], local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    enc = tok([r["prompt"] for r in prompts], padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")
    ids = enc.input_ids
    mask = enc.attention_mask
    pos = mask.long().cumsum(-1) - 1
    pos.masked_fill_(mask == 0, 0)
    cachep = cacheq = None
    b = len(prompts)
    alive = torch.ones(b, device="cuda", dtype=torch.bool)
    sequences = [[] for _ in prompts]
    finished = [False] * b
    arrays = {}
    proposal_reject = torch.zeros(b, device="cuda")
    empty = torch.zeros(b, device="cuda")
    gp = torch.Generator(device="cuda").manual_seed(seed)
    gr = torch.Generator(device="cuda").manual_seed(seed + 1000003)
    wp = student.lm_head.weight.detach().float()
    wq = teacher.lm_head.weight.detach().float()
    for t in range(cap):
        reject = torch.zeros(b, device="cuda", dtype=torch.bool)
        no = reject.clone()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            op = student.model(
                input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cachep, use_cache=True
            )
            oq = teacher.model(
                input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cacheq, use_cache=True
            )
        cachep = op.past_key_values
        cacheq = oq.past_key_values
        zp = torch.nn.functional.linear(op.last_hidden_state[:, -1].float(), wp)
        zt = torch.nn.functional.linear(oq.last_hidden_state[:, -1].float(), wq)
        prob = zp.softmax(-1)
        v = prob.shape[-1]
        if mode == "teacher":
            action = torch.multinomial(zt[:, :v].softmax(-1), 1, generator=gp).squeeze(1)
        else:
            action = torch.multinomial(prob, 1, generator=gp).squeeze(1)
            if mode in ["intersection", "topk"]:
                si = zp.topk(a.k, -1).indices
                ti = zt.topk(a.k, -1).indices
                good = (
                    (si[:, :, None] == ti[:, None, :]).any(-1)
                    if mode == "intersection"
                    else torch.ones_like(si, dtype=torch.bool)
                )
                no = ~good.any(-1)
                empty += no * alive
                good = torch.where(no[:, None], torch.ones_like(good), good)
                accept = ((si == action[:, None]) & good).any(-1)
                reject = ~accept
                proposal_reject += reject * alive
                probs = prob.gather(1, si) * good
                replacement = si.gather(1, torch.multinomial(probs, 1, generator=gr)).squeeze(1)
                action = torch.where(reject, replacement, action)
        values = vectors(zp, zt, action)
        values["alive"] = torch.ones_like(action, dtype=torch.float32)
        values["proposal_rejected"] = reject.float()
        values["empty_intersection"] = no.float()
        for name, val in values.items():
            if name not in arrays:
                arrays[name] = torch.zeros((b, cap), device="cuda")
            arrays[name][:, t] = val * alive
        selected = action.tolist()
        active = alive.tolist()
        for i, (token, on) in enumerate(zip(selected, active)):
            if on:
                sequences[i].append(token)
                finished[i] = token == tok.eos_token_id
        alive = alive & (action != tok.eos_token_id)
        if not alive.any():
            break
        ids = torch.where(alive, action, torch.full_like(action, tok.eos_token_id))[:, None]
        mask = torch.cat([mask, torch.ones((b, 1), device="cuda", dtype=mask.dtype)], -1)
        pos = mask.sum(-1, keepdim=True) - 1
        if (t + 1) % 256 == 0:
            print("decode", mode, t + 1, "/", cap, flush=True)
    cpu = {k: v.cpu() for k, v in arrays.items()}
    rejects = proposal_reject.tolist()
    empties = empty.tolist()
    records = []
    for i, r in enumerate(prompts):
        bands = {}
        for name, lo, hi in [
            ("all", 0, cap),
            ("early", 0, min(256, cap)),
            ("middle", 256, min(512, cap)),
            ("late", 512, cap),
        ]:
            if hi <= lo:
                continue
            bands[name] = {key: float(value[i, lo:hi].sum()) for key, value in cpu.items()}
        records.append(
            dict(
                r,
                response_ids=sequences[i],
                output=tok.decode(sequences[i], skip_special_tokens=False),
                finish_reason="stop" if finished[i] else "length",
                mode=mode,
                seed=seed,
                cap=cap,
                stats=bands,
                proposal_rejections=rejects[i],
                empty_intersections=empties[i],
                token_diagnostics={
                    key: cpu[key][i, : len(sequences[i])].tolist()
                    for key in ["rkl", "hp", "hq", "overlap16", "common_p16", "proposal_rejected"]
                },
                token_categories={
                    str(k): torch.stack([cpu[f"category{k}_{c}"][i, : len(sequences[i])] for c in range(4)])
                    .argmax(0)
                    .tolist()
                    for k in KS
                },
            )
        )
    del op, oq, cachep, cacheq, wp, wq, arrays, cpu
    torch.cuda.empty_cache()
    return records


@torch.no_grad()
def reference(a):
    prompts = [r for r in read(a.out / "prompts.json") if r["split"] == "heldout"]
    tok = AutoTokenizer.from_pretrained(a.models / NAMES[a.source], local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = load(a, a.source)
    dest = a.out / f"reference-{a.source}.json"
    records = []
    for start in range(0, len(prompts), 32):
        batch = prompts[start : start + 32]
        torch.manual_seed(73100 + start)
        enc = tok([r["prompt"] for r in batch], padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")
        out = model.generate(
            **enc,
            max_new_tokens=a.cap,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            use_cache=True,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        for r, seq in zip(batch, out[:, enc.input_ids.shape[1] :].tolist()):
            stop = tok.eos_token_id in seq
            if stop:
                seq = seq[: seq.index(tok.eos_token_id) + 1]
            records.append(
                dict(r, source=a.source, response_ids=seq, finish_reason="stop" if stop else "length", cap=a.cap)
            )
        dump(dest, records)
        print("reference", a.source, len(records), flush=True)


def refs(a):
    panels = [read(a.out / f"reference-{s}.json") for s in NAMES]
    assert all(len(panel) == 96 for panel in panels), "Reference generation must finish before training/evaluation"
    return [r for panel in panels for r in panel]


@torch.no_grad()
def teacher_hidden(a, teacher, r, cache=True):
    key = f"{r.get('source', 'train')}-{r['id']}"
    path = a.cache / a.teacher / (key + ".pt")
    if cache and path.exists():
        return torch.load(path, weights_only=True).to("cuda")
    h = hidden(teacher, r["prompt_ids"] + r["response_ids"][:-1])[len(r["prompt_ids"]) - 1 :].detach()
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(f".tmp-{os.getpid()}")
        torch.save(h.cpu(), temp)
        temp.replace(path)
    return h


@torch.no_grad()
def fixed_eval(a, student, teacher, step, dest, records=None):
    output = []
    wq = teacher.lm_head.weight.float()
    wp = student.lm_head.weight
    records = refs(a) if records is None else records
    for i, r in enumerate(records):
        ht = teacher_hidden(a, teacher, r)
        hs = hidden(student, r["prompt_ids"] + r["response_ids"][:-1])[len(r["prompt_ids"]) - 1 :]
        allvals = {}
        for start in range(0, len(hs), 128):
            sp = torch.nn.functional.linear(hs[start : start + 128].float(), wp)
            tq = torch.nn.functional.linear(ht[start : start + 128].float(), wq)
            actions = torch.tensor(r["response_ids"][start : start + len(sp)], device="cuda")
            vals = vectors(sp, tq, actions)
            for k, v in vals.items():
                allvals.setdefault(k, []).extend(v.tolist())
        bands = {}
        for name, lo, hi in [("all", 0, a.cap), ("early", 0, 256), ("middle", 256, 512), ("late", 512, a.cap)]:
            n = max(0, min(hi, len(hs)) - lo)
            bands[name] = {"alive": n, **{k: sum(v[lo:hi]) for k, v in allvals.items()}}
        output.append(
            dict(step=step, prompt_id=r["id"], domain=r["domain"], source=r["source"], length=len(hs), stats=bands)
        )
        del ht, hs, allvals, sp, tq
        if (i + 1) % 64 == 0:
            print("fixed_eval", step, i + 1, "/", len(records), flush=True)
    dump(dest, output)
    del wq
    torch.cuda.empty_cache()
    return output


@torch.no_grad()
def fresh_eval(a, student, teacher, step, dest):
    prompts = [r for r in read(a.out / "prompts.json") if r["split"] == "heldout" and r["domain"] == "math"][:32]
    result = []
    for seed in [91730, 91731]:
        rows = decode(a, student, teacher, prompts, "student", seed)
        result.extend([dict(r, step=step) for r in rows])
        dump(dest, result)
    return result


def run(a):
    arm = f"{a.teacher}-{a.mode}-k{a.k}"
    dest = a.out / arm
    dest.mkdir(exist_ok=True)
    student = load(a, "student", student=True)
    teacher = load(a, a.teacher)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-6, betas=(0.9, 0.999), weight_decay=0.01, foreach=True)
    protocol = read(a.out / "protocol.json") | dict(
        teacher=a.teacher, mode=a.mode, k=a.k, arm=arm, parameter_count=sum(p.numel() for p in student.parameters())
    )
    dump(dest / "protocol.json", protocol)
    training = []
    started = time.time()
    # Baseline is independent of the training arm. A teacher-specific lock avoids
    # concurrent writers when several Polaris arms start together.
    baseline = a.out / f"baseline-{a.teacher}"
    lock = a.out / f".baseline-{a.teacher}.lock"
    own = False
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        own = True
    except FileExistsError:
        pass
    if own:
        baseline.mkdir(exist_ok=True)
        fixed_eval(a, student, teacher, 0, baseline / "fixed.json")
        fresh_eval(a, student, teacher, 0, baseline / "fresh.json")
        (baseline / "done").write_text("complete\n")
    else:
        print("Independent initial baseline is being supplied by another arm", a.teacher, flush=True)
    prompts = [r for r in read(a.out / "prompts.json") if r["split"] == "train"]
    step = 0
    for outer in range(4):
        batch = prompts[32 * outer : 32 * (outer + 1)]
        rows = decode(a, student, teacher, batch, a.mode, 88000 + outer)
        dump(dest / f"rollout-{outer + 1}.json", rows)
        with torch.no_grad():
            hts = [teacher_hidden(a, teacher, r, cache=False) for r in rows]
        teacherhead = teacher.lm_head.weight.float()
        total = sum(len(r["response_ids"]) for r in rows)
        for reuse in range(16):
            optimizer.zero_grad(set_to_none=True)
            value = 0.0
            for r, ht in zip(rows, hts):
                hs = hidden(student, r["prompt_ids"] + r["response_ids"][:-1])[len(r["prompt_ids"]) - 1 :]
                losses = []
                for start in range(0, len(hs), 128):
                    sp = torch.nn.functional.linear(hs[start : start + 128].float(), student.lm_head.weight)
                    with torch.no_grad():
                        tq = torch.nn.functional.linear(ht[start : start + 128].float(), teacherhead).log_softmax(-1)[
                            :, : sp.shape[-1]
                        ]
                    lp = sp.log_softmax(-1)
                    losses.append((lp.exp() * (lp - tq)).sum() / total)
                loss = sum(losses)
                loss.backward()
                value += float(loss.detach())
                del hs, sp, tq, lp, loss, losses
            norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            assert torch.isfinite(norm) and math.isfinite(value)
            optimizer.step()
            step += 1
            training.append(
                dict(
                    step=step,
                    outer=outer + 1,
                    reuse=reuse + 1,
                    pre_update_kl=value,
                    grad_norm=float(norm),
                    valid_tokens=total,
                    seconds=time.time() - started,
                    peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                )
            )
            dump(dest / "training.json", training)
            print("update", arm, training[-1], flush=True)
        optimizer.zero_grad(set_to_none=True)
        del hts, teacherhead
        torch.cuda.empty_cache()
        fixed_eval(a, student, teacher, step, dest / f"fixed-{step}.json")
        if step in [32, 64]:
            fresh_eval(a, student, teacher, step, dest / f"fresh-{step}.json")
    # Endpoint is a task-specific artifact, never an overwrite of the model assets.
    if a.save:
        checkpoint = a.cache / "checkpoints" / arm
        checkpoint.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(checkpoint, max_shard_size="4GB")
        AutoTokenizer.from_pretrained(a.models / NAMES["student"], local_files_only=True).save_pretrained(checkpoint)
    while not (baseline / "done").exists():
        print("waiting for baseline completion", a.teacher, flush=True)
        time.sleep(10)
    dump(dest / "complete.json", dict(updates=step, seconds=time.time() - started, finished=True))


@torch.no_grad()
def task_eval(a):
    if a.checkpoint:
        student = (
            AutoModelForCausalLM.from_pretrained(
                a.checkpoint, dtype=torch.float32, attn_implementation="flash_attention_2", local_files_only=True
            )
            .to("cuda")
            .eval()
        )
    else:
        student = load(a, "student", student=True)
    teacher = load(a, a.teacher)
    rows = decode(a, student, teacher, read(a.out / "task-prompts.json"), "student", 44719, cap=4096)
    dump(a.out / f"task-{a.tag}.json", rows)
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
    from verl.utils.reward_score.one_shot_opd.math import reward_func

    for r in rows:
        r["grade"] = reward_func("MATH-500", r["output"], r["ground_truth"])
    dump(a.out / f"task-{a.tag}.json", rows)


def selection(a):
    a.teacher = "polaris"
    dest = a.out / f"selection-{a.selection}"
    dest.mkdir(exist_ok=True)
    rows = read(a.out / "polaris-student-k16/rollout-1.json")
    student = load(a, "student", student=True)
    teacher = load(a, "polaris")
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-6, betas=(0.9, 0.999), weight_decay=0.01, foreach=True)
    masks = []
    description = []
    for r in rows:
        shuffled = list(range(len(r["response_ids"])))
        random.Random(8120 + r["id"]).shuffle(shuffled)
        if a.selection == "high":
            shuffled.sort(key=lambda i: r["token_diagnostics"]["overlap16"][i], reverse=True)
        elif a.selection == "low":
            shuffled.sort(key=lambda i: r["token_diagnostics"]["overlap16"][i])
        chosen = sorted(shuffled[: min(256, len(shuffled))])
        masks.append(chosen)
        description.append(
            dict(
                id=r["id"],
                positions=chosen,
                **{
                    k: sum(r["token_diagnostics"][k][i] for i in chosen) / len(chosen)
                    for k in ["overlap16", "rkl", "hp", "hq"]
                },
            )
        )
    dump(
        dest / "protocol.json",
        dict(
            teacher="polaris",
            selection=a.selection,
            updates=16,
            batch=32,
            source="same initial student rollout, outer 1",
            selected_per_response=256,
            tie_break="seeded shuffle then stable sort",
            lr=1e-6,
            clip=1.0,
            fixed_full_input=True,
        ),
    )
    dump(dest / "masks.json", description)
    with torch.no_grad():
        hts = [teacher_hidden(a, teacher, r, cache=False) for r in rows]
    wq = teacher.lm_head.weight.float()
    total = sum(map(len, masks))
    history = []
    started = time.time()
    for step in range(1, 17):
        optimizer.zero_grad(set_to_none=True)
        value = 0.0
        for r, ht, positions in zip(rows, hts, masks):
            # Same entire forward input for every selection; only loss states differ.
            hs = hidden(student, r["prompt_ids"] + r["response_ids"][:-1])[len(r["prompt_ids"]) - 1 :]
            index = torch.tensor(positions, device="cuda")
            hs = hs[index]
            target = ht[index]
            losses = []
            for start in range(0, len(hs), 128):
                sp = torch.nn.functional.linear(hs[start : start + 128].float(), student.lm_head.weight)
                with torch.no_grad():
                    lq = torch.nn.functional.linear(target[start : start + 128].float(), wq).log_softmax(-1)[
                        :, : sp.shape[-1]
                    ]
                lp = sp.log_softmax(-1)
                losses.append((lp.exp() * (lp - lq)).sum() / total)
            loss = sum(losses)
            loss.backward()
            value += float(loss.detach())
            del hs, target, sp, lq, lp, loss, losses
        norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        assert torch.isfinite(norm)
        before = [p.detach().clone() for p in student.parameters()]
        optimizer.step()
        with torch.no_grad():
            squared = torch.zeros((), device="cuda")
            for old, new in zip(before, student.parameters()):
                squared += (new - old).square().sum()
        displacement = float(squared.sqrt())
        del before, squared
        history.append(
            dict(
                step=step,
                loss=value,
                grad_norm=float(norm),
                parameter_step_norm=displacement,
                tokens=total,
                seconds=time.time() - started,
            )
        )
        dump(dest / "training.json", history)
        print("selection", a.selection, history[-1], flush=True)
    optimizer.zero_grad(set_to_none=True)
    del hts, wq
    torch.cuda.empty_cache()
    fixed_eval(a, student, teacher, 16, dest / "fixed-16.json")
    fresh_eval(a, student, teacher, 16, dest / "fresh-16.json")
    dump(dest / "complete.json", dict(updates=16, finished=True))


@torch.no_grad()
def decompose(a):
    out = a.out / "decomposition"
    out.mkdir(exist_ok=True)
    for target, arms in [
        ("justrl", ["justrl-student-k16"]),
        ("polaris", ["polaris-student-k16", "polaris-teacher-k16", "polaris-intersection-k16", "polaris-topk-k16"]),
    ]:
        a.teacher = target
        teacher = load(a, target)
        student = load(a, "student", student=True)
        old = [dict(r, source=f"old-{r['seed']}") for r in read(a.out / f"baseline-{target}/fresh.json")]
        fixed_eval(a, student, teacher, 0, out / f"{target}-oldmodel-oldstates.json", old)
        for arm in arms:
            new = [dict(r, source=f"new-{arm}-{r['seed']}") for r in read(a.out / arm / "fresh-64.json")]
            fixed_eval(a, student, teacher, 0, out / f"{arm}-oldmodel-newstates.json", new)
        del student
        gc.collect()
        torch.cuda.empty_cache()
        for arm in arms:
            student = (
                AutoModelForCausalLM.from_pretrained(
                    a.cache / "checkpoints" / arm,
                    dtype=torch.float32,
                    attn_implementation="flash_attention_2",
                    local_files_only=True,
                )
                .to("cuda")
                .eval()
            )
            new = [dict(r, source=f"new-{arm}-{r['seed']}") for r in read(a.out / arm / "fresh-64.json")]
            fixed_eval(a, student, teacher, 64, out / f"{arm}-newmodel-oldstates.json", old)
            fixed_eval(a, student, teacher, 64, out / f"{arm}-newmodel-newstates.json", new)
            del student
            gc.collect()
            torch.cuda.empty_cache()
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
    dump(out / "complete.json", dict(finished=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "reference", "run", "task_eval", "selection", "decompose"])
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--source", choices=list(NAMES), default="student")
    p.add_argument("--teacher", choices=["justrl", "polaris"], default="polaris")
    p.add_argument("--mode", choices=["student", "teacher", "intersection", "topk"], default="student")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--cap", type=int, default=1024)
    p.add_argument("--save", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--tag", default="initial")
    p.add_argument("--selection", choices=["high", "low", "random"], default="high")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    a.cache.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(2026102702)
    if a.stage != "prepare":
        torch.cuda.set_per_process_memory_fraction(0.88)
    globals()[a.stage](a)


if __name__ == "__main__":
    main()
