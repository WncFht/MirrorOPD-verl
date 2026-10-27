#!/usr/bin/env python3

"""Frozen-state probe primitives: native full-vocabulary metrics and joint decoding.
Kept independent of earlier study artifact directories.
"""

import json
import math
import os

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
