#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared native-precision generation/cache helpers for the filter hypothesis."""

import torch
from transformers import AutoTokenizer

from examples.mirroropd.research.backends import hf_native as core
from examples.mirroropd.research.core.cache import HiddenCache, model_identity


@torch.no_grad()
def cache_hidden(cache, namespace, model, rows):
    store = HiddenCache(cache, namespace, model_identity(model))
    for index, row in enumerate(rows, 1):
        store.get(
            row,
            lambda row=row: core.hidden(model, row["prompt_ids"] + row["response_ids"][:-1])[
                len(row["prompt_ids"]) - 1 :
            ],
        )
        if index % 128 == 0:
            print("CACHE", namespace, index, "/", len(rows), flush=True)


def load_hidden(cache, namespace, rows, *, model):
    store = HiddenCache(cache, namespace, model_identity(model))
    result = {}
    for row in rows:
        key = (row["source"], row["id"])
        if key in result:
            raise ValueError("This historical filter protocol expects one rollout per source/prompt")
        result[key] = store.load(row)
    return result


@torch.no_grad()
def generate(a, model, prompts, seed, cap, vocab_limit=None):
    tokenizer = AutoTokenizer.from_pretrained(a.models / core.NAMES["student"], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    encoded = tokenizer([r["prompt"] for r in prompts], padding=True, return_tensors="pt", add_special_tokens=False).to(
        "cuda"
    )
    ids, mask = encoded.input_ids, encoded.attention_mask
    positions = mask.long().cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    past = None
    active = torch.ones(len(prompts), dtype=torch.bool, device="cuda")
    output = [[] for _ in prompts]
    stopped = [False] * len(prompts)
    head = model.lm_head.weight.detach().float()
    rng = torch.Generator(device="cuda").manual_seed(seed)
    for step in range(cap):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model.model(
                input_ids=ids, attention_mask=mask, position_ids=positions, past_key_values=past, use_cache=True
            )
        past = result.past_key_values
        logits = torch.nn.functional.linear(result.last_hidden_state[:, -1].float(), head)
        if vocab_limit is not None:
            logits = logits[:, :vocab_limit]
        action = torch.multinomial(logits.softmax(-1), 1, generator=rng).squeeze(1)
        for i, (token, on) in enumerate(zip(action.tolist(), active.tolist(), strict=False)):
            if on:
                output[i].append(token)
                stopped[i] = token == tokenizer.eos_token_id
        active = active & (action != tokenizer.eos_token_id)
        if not active.any():
            break
        ids = torch.where(active, action, torch.full_like(action, tokenizer.eos_token_id))[:, None]
        mask = torch.cat([mask, torch.ones_like(ids)], -1)
        positions = mask.sum(-1, keepdim=True) - 1
        if (step + 1) % 1024 == 0:
            print("GENERATE", len(prompts), step + 1, "/", cap, flush=True)
    rows = [
        dict(
            r,
            response_ids=output[i],
            output=tokenizer.decode(output[i], skip_special_tokens=False),
            finish_reason="stop" if stopped[i] else "length",
            cap=cap,
            seed=seed,
        )
        for i, r in enumerate(prompts)
    ]
    del past, result, head, logits, encoded
    torch.cuda.empty_cache()
    return rows


def grade_math(rows, repo):
    import sys

    sys.path.insert(0, str(repo))
    from verl.utils.reward_score.one_shot_opd.math import reward_func

    for r in rows:
        r["grade"] = reward_func("MATH-500", r["output"], r["ground_truth"])
    return rows
