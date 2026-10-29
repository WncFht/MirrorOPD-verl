# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact native-support RKL VJP and batch-resident frozen teacher probabilities."""

import gc

import torch

from ...common.metrics import fixed_metrics
from .native import bank, hidden, load
from .round2_common import CACHE, GROUPS, MODELS, OLD_CACHE, dump, model_path, read


def panel():
    return [r for r in bank("r1") if r["group"] in GROUPS]


def training_mode(model):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()


def teacher_cache(teacher, rows):
    old = OLD_CACHE / "teachers" / teacher / "r1"
    directory = old if (old / "complete.json").exists() else CACHE / "teachers" / teacher / "r1"
    if not (directory / "complete.json").exists():
        directory.mkdir(parents=True, exist_ok=True)
        model = load(teacher, path=model_path(teacher))
        torch.save(model.lm_head.weight.detach().cpu(), directory / "head.pt")
        with torch.no_grad():
            for i, row in enumerate(rows):
                path = directory / f"{row['id']:04d}.pt"
                if not path.exists():
                    torch.save(
                        dict(
                            prompt_ids=row["prompt_ids"],
                            response_ids=row["response_ids"],
                            teacher=MODELS[teacher],
                            source=row["source"],
                            hidden=hidden(model, row).cpu(),
                        ),
                        path,
                    )
                if i % 32 == 0:
                    print("CACHE", teacher, i, len(rows), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        dump(directory / "complete.json", dict(teacher=MODELS[teacher], rows=len(rows)))
    assert read(directory / "complete.json")["teacher"] == list(MODELS[teacher])
    states = {}
    for row in rows:
        payload = torch.load(directory / f"{row['id']:04d}.pt", map_location="cpu", weights_only=True)
        assert tuple(payload["teacher"]) == MODELS[teacher]
        assert payload["prompt_ids"] == row["prompt_ids"]
        assert payload["response_ids"] == row["response_ids"]
        assert payload["source"] == row["source"]
        states[row["id"]] = payload["hidden"]
    head = torch.load(directory / "head.pt", map_location="cuda", weights_only=True).float()
    return head, states


@torch.no_grad()
def probabilities(head, states, rows, chunk=256):
    """Keep the teacher's complete softmax normalizer, including extra vocabulary."""
    result = {}
    for row in rows:
        h = states[row["id"]].to(device=head.device, dtype=torch.float32)
        q = torch.empty((len(h), len(head)), dtype=torch.float32, device=head.device)
        for start in range(0, len(h), chunk):
            q[start : start + chunk] = torch.nn.functional.linear(h[start : start + chunk], head).log_softmax(-1)
        result[row["id"]] = q
    return result


@torch.no_grad()
def head_vjp(h, weight, lq, denominator):
    """d KL(p||q)/dz = p * (log p - log q - KL(p||q))."""
    lp = torch.nn.functional.linear(h, weight).log_softmax(-1)
    p = lp.exp()
    difference = lp - lq[:, : len(weight)]
    rkl = (p * difference).sum(-1)
    dz = p * (difference - rkl[:, None]) / denominator
    return rkl.sum(dtype=torch.float64) / denominator, dz


def backward(model, qcache, rows, chunk=256):
    """Head VJP in FP32, then one backbone backward per sequence; batch16 stays fixed."""
    weight = model.lm_head.weight
    if weight.grad is None:
        weight.grad = torch.zeros_like(weight)
    value = torch.zeros((), device=weight.device, dtype=torch.float64)
    for row in rows:
        hp = hidden(model, row)
        h = hp.detach().float()
        dh = torch.empty_like(h)
        for start in range(0, len(h), chunk):
            stop = start + chunk
            loss, dz = head_vjp(h[start:stop], weight, qcache[row["id"]][start:stop], 16 * 1024)
            with torch.no_grad():
                weight.grad.addmm_(dz.T, h[start:stop])
                dh[start:stop] = dz @ weight
                value += loss
        hp.backward(dh.to(hp.dtype))
    return float(value)


@torch.no_grad()
def score(model, head, states, rows, step, chunk=256):
    results = []
    bands_spec = [("all", 0, 1024), ("early", 0, 256), ("middle", 256, 512), ("late", 512, 1024)]
    for index, row in enumerate(rows, 1):
        hp = hidden(model, row).float()
        hq = states[row["id"]].to(device=head.device, dtype=torch.float32)
        sums = None
        for start in range(0, len(hp), chunk):
            sp = torch.nn.functional.linear(hp[start : start + chunk], model.lm_head.weight)
            tq = torch.nn.functional.linear(hq[start : start + chunk], head)
            values = fixed_metrics(sp, tq)
            values["extra_teacher_mass"] = tq.softmax(-1)[:, sp.shape[-1] :].sum(-1)
            if sums is None:
                keys = list(values)
                sums = torch.zeros((4, len(keys)), dtype=torch.float64, device=head.device)
            for band, (_, lo, hi) in enumerate(bands_spec):
                left, right = max(lo - start, 0), min(hi - start, len(sp))
                if right > left:
                    sums[band] += torch.stack([values[k][left:right].sum(dtype=torch.float64) for k in keys])
        bands = {
            name: dict(tokens=max(0, min(len(hp), hi) - lo), **dict(zip(keys, vector, strict=True)))
            for (name, lo, hi), vector in zip(bands_spec, sums.tolist(), strict=True)
        }
        results.append(
            dict(
                id=row["id"],
                group=row["group"],
                domain=row["domain"],
                split=row["split"],
                step=step,
                length=len(hp),
                stats=bands,
                source=row["source"],
            )
        )
        if index % 64 == 0:
            print("SCORE", step, index, len(rows), flush=True)
    return results


def hidden_batch(model, rows):
    """Left-pad independent trajectories; explicit masks preserve positional identities."""
    sequences = [r["prompt_ids"] + r["response_ids"][:-1] for r in rows]
    width = max(map(len, sequences))
    device = model.model.embed_tokens.weight.device
    ids = torch.tensor([[151643] * (width - len(s)) + s for s in sequences], device=device)
    mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in sequences], device=device)
    positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.model(
            input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False
        ).last_hidden_state
    return torch.cat(
        [
            output[i, width - len(s) + len(r["prompt_ids"]) - 1 :]
            for i, (s, r) in enumerate(zip(sequences, rows, strict=True))
        ]
    )


def backward_batched(model, qcache, rows, chunk=512, microbatch=4):
    """Optional next-iteration engine; separate from the six frozen scientific arms."""
    weight = model.lm_head.weight
    if weight.grad is None:
        weight.grad = torch.zeros_like(weight)
    value = torch.zeros((), device=weight.device, dtype=torch.float64)
    for index in range(0, len(rows), microbatch):
        part = rows[index : index + microbatch]
        hp = hidden_batch(model, part)
        h = hp.detach().float()
        dh = torch.empty_like(h)
        offset = 0
        for row in part:
            length = len(row["response_ids"])
            for start in range(0, length, chunk):
                stop = min(start + chunk, length)
                begin, end = offset + start, offset + stop
                loss, dz = head_vjp(h[begin:end], weight, qcache[row["id"]][start:stop], 16 * 1024)
                with torch.no_grad():
                    weight.grad.addmm_(dz.T, h[begin:end])
                    dh[begin:end] = dz @ weight
                    value += loss
            offset += length
        assert offset == len(hp)
        hp.backward(dh.to(hp.dtype))
    return float(value)
