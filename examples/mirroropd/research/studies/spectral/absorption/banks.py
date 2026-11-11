# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Generate and score native-vocabulary fixed-prefix banks."""

import json
import math

import torch

from examples.mirroropd.research.core.io import dump

from ...teacher_difference.runtime import collect_memory, load_model, response_hidden


def generated_bank(context, model, tokenizer, prompts, origin):
    rows = []
    eos = tokenizer.eos_token_id
    for start in range(0, len(prompts), 4):
        context.check_stop()
        batch = prompts[start : start + 4]
        tokens = [r["prompt_ids"] for r in batch]
        width = max(map(len, tokens))
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
        ids = torch.tensor([[pad] * (width - len(x)) + x for x in tokens], device="cuda")
        mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in tokens], device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            answer = (
                model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=pad,
                    eos_token_id=eos,
                    use_cache=True,
                )[:, width:]
                .cpu()
                .tolist()
            )
        for row, response in zip(batch, answer, strict=True):
            if eos is not None and eos in response:
                response = response[: response.index(eos) + 1]
            rows.append(row | dict(response_ids=response, origin=origin, eos_token_id=eos))
    return rows


def collect_reference(context, path, banks):
    model = load_model(path, trainable=True, device="cuda")
    model.eval().requires_grad_(False)
    hidden = {}
    with torch.no_grad():
        for origin, rows in banks.items():
            for row in rows:
                context.check_stop()
                hidden[origin, row["row_id"]] = response_hidden(model, row)
    head = model.lm_head.weight.detach().cpu().clone()
    del model
    collect_memory()
    return hidden, head


def score(context, model, banks, reference, head, destination, condition):
    if destination.exists():
        return json.loads(destination.read_text())
    values = []
    with torch.no_grad():
        for origin, rows in banks.items():
            for row in rows:
                context.check_stop()
                hs = response_hidden(model, row).cuda()
                ht = reference[origin, row["row_id"]].cuda()
                accum = dict(rkl=0.0, fkl=0.0, js=0.0, student_entropy=0.0, teacher_entropy=0.0)
                per_token = {name: [] for name in accum}
                for first in range(0, len(hs), 16):
                    ls = torch.nn.functional.linear(hs[first : first + 16], model.lm_head.weight).log_softmax(-1)
                    lt = torch.nn.functional.linear(ht[first : first + 16], head).log_softmax(-1)
                    ps, pt = ls.exp(), lt.exp()
                    lm = torch.logaddexp(ls, lt) - math.log(2)
                    metrics = dict(
                        rkl=(ps * (ls - lt)).sum(-1),
                        fkl=(pt * (lt - ls)).sum(-1),
                        js=((ps * (ls - lm)).sum(-1) + (pt * (lt - lm)).sum(-1)) / 2,
                        student_entropy=-(ps * ls).sum(-1),
                        teacher_entropy=-(pt * lt).sum(-1),
                    )
                    for name, v in metrics.items():
                        accum[name] += float(v.double().sum())
                        per_token[name].extend(v.cpu().tolist())
                values.append(
                    dict(
                        origin=origin,
                        row_id=row["row_id"],
                        split=row["split"],
                        domain=row["domain"],
                        tokens=len(hs),
                        **{name: value / 128 for name, value in accum.items()},
                        per_actual_token={name: value / len(hs) for name, value in accum.items()},
                        per_token=per_token,
                    )
                )
                dump(destination.with_suffix(".progress.json"), dict(condition=condition, completed=len(values)))
    result = dict(
        condition=condition, denominator="fixed H128 per question; actual-token values also saved", rows=values
    )
    dump(destination, result)
    return result
