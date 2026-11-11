# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Finite-pool sampling and exact chunked CE with a merged stopping event."""

import math
import random
import statistics

ARMS = ("RAW-C", "GRAPE", "RSR", "RATIO")


def probabilities(candidates, arm, alpha, *, top=4):
    if arm not in ARMS or len(candidates) < top:
        raise ValueError("Unknown arm or insufficient common candidates")
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0,1]")
    if arm == "RATIO":
        logits = [alpha * (c["base"]["sequence_logprob"] - c["expert"]["sequence_logprob"]) for c in candidates]
        if not all(map(math.isfinite, logits)):
            raise ValueError("Nonfinite density ratio")
        weights = [math.exp(v - max(logits)) for v in logits]
    elif arm in ("GRAPE", "RSR"):
        key = "mean_logprob" if arm == "GRAPE" else "rsr"
        indices = sorted(range(len(candidates)), key=lambda j: candidates[j]["base"][key], reverse=arm == "GRAPE")
        chosen = set(indices[:top])
        weights = [float(j in chosen) for j in range(len(candidates))]
    else:
        weights = [1.0] * len(candidates)
    return [w / sum(weights) for w in weights]


def weight_diagnostics(p):
    return dict(ess=1 / sum(v * v for v in p), max_weight=max(p), positive_candidates=sum(v > 0 for v in p))


def calibrate_alpha(groups, grid=(0.01, 0.03, 0.1, 0.3, 1.0)):
    if not groups:
        raise ValueError("Empty calibration pool")
    records = []
    for alpha in grid:
        med = statistics.median(weight_diagnostics(probabilities(c, "RATIO", alpha))["ess"] / len(c) for c in groups)
        records.append(dict(alpha=alpha, median_ess_fraction=med))
    return min(records, key=lambda r: (abs(r["median_ess_fraction"] - 0.5), r["alpha"]))["alpha"], records


def selection_table(groups, seed, alpha, *, epochs=4):
    """Common question order and common uniforms across arms, independent of model RNG."""
    rng = random.Random(seed)
    indices = list(range(len(groups)))
    for epoch in range(1, epochs + 1):
        rng.shuffle(indices)
        for i in indices:
            uniform = rng.random()
            choices = {}
            for arm in ARMS:
                p = probabilities(groups[i]["candidates"], arm, alpha)
                cumulative = 0.0
                selected = len(p) - 1
                for j, weight in enumerate(p):
                    cumulative += weight
                    if uniform < cumulative:
                        selected = j
                        break
                choices[arm] = groups[i]["candidates"][selected]["candidate_id"]
            yield dict(epoch=epoch, question_id=groups[i]["id"], uniform=uniform, choices=choices)


def validate_content(prompt, content, end_ids, *, max_prompt=4096, max_response=24576, max_length=32768):
    if not prompt or len(prompt) > max_prompt:
        raise ValueError("Empty or over-budget prompt")
    if not content or any(t in end_ids for t in content):
        raise ValueError("Empty body or internal stop token")
    # The actual stopping draw consumes one generation position.
    if len(content) + 1 > max_response or len(prompt) + len(content) + 1 > max_length:
        raise ValueError("Complete response exceeds budget; truncation is forbidden")


def log_probabilities(states, weight, support_size=None):
    import torch.nn.functional as F

    logits = F.linear(states.float(), weight.float())
    if support_size is not None:
        if not 0 < support_size <= logits.shape[-1]:
            raise ValueError("Invalid common vocabulary support")
        logits[:, support_size:] = -float("inf")
    return logits.log_softmax(-1)


def merged_statistics(states, weight, content, end_ids, *, chunk=64, rank_clip=100, support_size=None):
    """FP32 head, FP64 sequence accumulation; ranks clip with strict-greater tie semantics."""
    import torch

    if len(states) != len(content) + 1 or not content:
        raise ValueError("Expected one prediction state per content token plus END")
    values, ranks = [], {r: 0.0 for r in (10, rank_clip, 1000)}
    for first in range(0, len(content), chunk):
        lp = log_probabilities(states[first : min(first + chunk, len(content))], weight, support_size)
        target = torch.tensor(content[first : first + chunk], device=lp.device)
        selected = lp.gather(1, target[:, None]).squeeze(1)
        values.append(float(selected.double().sum()))
        rank = 1 + (lp > selected[:, None]).sum(-1)
        for r in ranks:
            ranks[r] += float(rank.clamp(max=r).double().sum())
    lp_end = log_probabilities(states[-1:], weight, support_size)[0]
    end = float(torch.logsumexp(lp_end[list(end_ids)], 0))
    total = math.fsum(values)
    return dict(
        content_logprob=total,
        end_logprob=end,
        sequence_logprob=total + end,
        mean_logprob=total / len(content),
        rsr=ranks[rank_clip] / max(-total, 1e-30),
        rsr_sensitivity={str(r): v / max(-total, 1e-30) for r, v in ranks.items()},
    )


def merged_ce_backward(model, value, end_ids, denominator, *, chunk=64, support_size=None):
    """Exact head VJP, including both tied embedding routes and merged-END derivatives."""
    import torch

    from examples.mirroropd.research.backends.hf_models import hidden

    if denominator <= 0 or not math.isfinite(denominator):
        raise ValueError("Positive fixed denominator required")
    weight = model.get_output_embeddings().weight
    if weight.dtype != torch.float32:
        raise ValueError("The training projection/master must remain FP32")
    content = value["content_ids"]
    row = dict(prompt_ids=value["prompt_ids"], response_ids=content + [end_ids[0]])
    states = hidden(model, [row], pad_id=end_ids[0])
    if len(states) != len(content) + 1:
        raise ValueError("Response alignment changed")
    detached = states.detach().float()
    if weight.grad is None:
        weight.grad = torch.zeros_like(weight)
    dh = torch.empty_like(detached)
    loss = 0.0
    for first in range(0, len(states), chunk):
        stop = min(first + chunk, len(states))
        with torch.no_grad():
            lp = log_probabilities(detached[first:stop], weight, support_size)
            dz = lp.exp()
            n_body = min(stop, len(content)) - first
            if n_body > 0:
                targets = torch.tensor(content[first : first + n_body], device=weight.device)
                loss -= float(lp[:n_body].gather(1, targets[:, None]).double().sum()) / denominator
                dz[:n_body].scatter_add_(1, targets[:, None], -torch.ones((n_body, 1), device=weight.device))
            if stop == len(states):
                terminal = lp[-1, list(end_ids)]
                loss -= float(torch.logsumexp(terminal, 0)) / denominator
                dz[-1, list(end_ids)] -= terminal.softmax(0)
            dz.div_(denominator)
            weight.grad.addmm_(dz.T, detached[first:stop])
            dh[first:stop] = dz @ weight
    states.backward(dh.to(states.dtype))
    return loss
