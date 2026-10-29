# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One native-support KL implementation for tied/untied heads and all study models."""

from pathlib import Path

import torch

from ...common.metrics import native_metrics
from .models import hidden


@torch.no_grad()
def head_vjp(states, weight, log_q, denominator):
    if log_q.shape[1] < weight.shape[0]:
        raise ValueError("Teacher support does not cover student support")
    log_p = torch.nn.functional.linear(states, weight).log_softmax(-1)
    p = log_p.exp()
    difference = log_p - log_q[:, : weight.shape[0]]
    rkl = (p * difference).sum(-1)
    dz = p * (difference - rkl[:, None]) / denominator
    return rkl.sum(dtype=torch.float64) / denominator, dz


@torch.no_grad()
def log_probabilities(head, teacher_states, rows, chunk):
    result = {}
    for row in rows:
        h = teacher_states[row["id"]].to(device=head.device, dtype=torch.float32)
        value = torch.empty((len(h), head.shape[0]), device=head.device, dtype=torch.float32)
        for start in range(0, len(h), chunk):
            value[start : start + chunk] = torch.nn.functional.linear(h[start : start + chunk], head).log_softmax(-1)
        result[row["id"]] = value
    return result


def backward(model, qcache, rows, training, execution, pad_id):
    weight = model.get_output_embeddings().weight
    # For tied models this is also the embedding gradient. Backbone autograd adds to it.
    if weight.grad is None:
        weight.grad = torch.zeros_like(weight)
    value = torch.zeros((), device=weight.device, dtype=torch.float64)
    for first in range(0, len(rows), execution.microbatch):
        subset = rows[first : first + execution.microbatch]
        hp = hidden(model, subset, pad_id=pad_id)
        states = hp.detach().float()
        dh = torch.empty_like(states)
        offset = 0
        for row in subset:
            length = len(row["response_ids"])
            for start in range(0, length, execution.token_chunk):
                stop = min(start + execution.token_chunk, length)
                left, right = offset + start, offset + stop
                loss, dz = head_vjp(states[left:right], weight, qcache[row["id"]][start:stop], training.denominator)
                with torch.no_grad():
                    weight.grad.addmm_(dz.T, states[left:right])
                    dh[left:right] = dz @ weight
                    value += loss
            offset += length
        assert offset == len(hp)
        hp.backward(dh.to(hp.dtype))
    return float(value)


def reference_backward(model, head, teacher_states, rows, training, pad_id):
    value = 0.0
    for row in rows:
        hp = hidden(model, [row], pad_id=pad_id)
        leaf = hp.detach().float().requires_grad_(True)
        hq = teacher_states[row["id"]].to(device=hp.device, dtype=torch.float32)
        for start in range(0, len(hp), 64):
            log_p = torch.nn.functional.linear(
                leaf[start : start + 64], model.get_output_embeddings().weight
            ).log_softmax(-1)
            with torch.no_grad():
                log_q = torch.nn.functional.linear(hq[start : start + 64], head).log_softmax(-1)
            loss = (log_p.exp() * (log_p - log_q[:, : log_p.shape[1]])).sum() / training.denominator
            value += float(loss.detach())
            loss.backward()
        hp.backward(leaf.grad.to(hp.dtype))
    return value


@torch.no_grad()
def score(model, head, teacher_states, rows, step, training, execution, pad_id):
    horizon = training.horizon
    bands = [
        ("all", 0, horizon),
        ("early", 0, horizon // 4),
        ("middle", horizon // 4, horizon // 2),
        ("late", horizon // 2, horizon),
    ]
    results = []
    for first in range(0, len(rows), execution.microbatch):
        subset = rows[first : first + execution.microbatch]
        states = hidden(model, subset, pad_id=pad_id).float()
        offset = 0
        for row in subset:
            length = len(row["response_ids"])
            hp = states[offset : offset + length]
            offset += length
            hq = teacher_states[row["id"]].to(device=head.device, dtype=torch.float32)
            sums = None
            for start in range(0, length, execution.token_chunk):
                stop = start + execution.token_chunk
                sp = torch.nn.functional.linear(hp[start:stop], model.get_output_embeddings().weight)
                tq = torch.nn.functional.linear(hq[start:stop], head)
                values = native_metrics(sp, tq)
                if sums is None:
                    keys = list(values)
                    sums = torch.zeros((len(bands), len(keys)), device=head.device, dtype=torch.float64)
                for index, (_, lo, hi) in enumerate(bands):
                    left, right = max(lo - start, 0), min(hi - start, len(sp))
                    if right > left:
                        sums[index] += torch.stack([values[k][left:right].sum(dtype=torch.float64) for k in keys])
            stats = {
                name: dict(tokens=max(0, min(length, hi) - lo), **dict(zip(keys, vector, strict=True)))
                for (name, lo, hi), vector in zip(bands, sums.tolist(), strict=True)
            }
            results.append(
                dict(
                    id=row["id"],
                    group=row["group"],
                    domain=row["domain"],
                    split=row["split"],
                    step=step,
                    length=length,
                    source=row["source"],
                    stats=stats,
                )
            )
        if len(results) % 64 == 0:
            print("SCORE", step, len(results), len(rows), flush=True)
    return results


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)
