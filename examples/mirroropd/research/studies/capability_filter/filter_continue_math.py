# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Continue selected capped errors without changing their saved response prefix.

Selection is an exploratory censoring diagnostic: initial-student-correct cases
that became incorrect AND capped after Polaris S0 training. This is not a new
unbiased longer-budget benchmark. Suffix RNG is newly seeded, not a replay of
an unsaved decoder RNG state.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.mirroropd.assets.paths import checkpoint_path, registered_model
from examples.mirroropd.research.backends import hf_native as core
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.capability_filter import filter_common as fc


@torch.no_grad()
def run(a):
    protocol = read(a.out / "protocol.json")["capped_answer_continuation"]
    selected = protocol["arms"][a.arm]
    target = a.out / "continuation" / (a.arm + ".json")
    assert not target.exists()
    original = read(a.out / a.arm / "task.json")
    rows = [r for r in original if r["id"] in selected["ids"]]
    assert len(rows) == len(selected["ids"]) and all(
        r["finish_reason"] == "length" and not r["grade"]["acc"] for r in rows
    )
    tokenizer = AutoTokenizer.from_pretrained(
        registered_model(core.NAMES["student"], root=a.models), local_files_only=True
    )
    eos = tokenizer.eos_token_id
    model = (
        AutoModelForCausalLM.from_pretrained(
            checkpoint_path(a.cache, "checkpoints") / a.arm / "step-128",
            dtype=torch.float32,
            attn_implementation="flash_attention_2",
            local_files_only=True,
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    inputs = [r["prompt_ids"] + r["response_ids"] for r in rows]
    width = max(map(len, inputs))
    batch = len(rows)
    assert width + protocol["additional_token_cap"] <= model.config.max_position_embeddings
    ids = torch.full((batch, width), eos, dtype=torch.long, device="cuda")
    mask = torch.zeros_like(ids)
    for i, tokens in enumerate(inputs):
        ids[i, -len(tokens) :] = torch.tensor(tokens, device="cuda")
        mask[i, -len(tokens) :] = 1
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    alive = torch.ones(batch, dtype=torch.bool, device="cuda")
    suffix = [[] for _ in rows]
    stopped = [False] * batch
    rng = torch.Generator(device="cuda").manual_seed(selected["seed"])
    past = None
    for step in range(protocol["additional_token_cap"]):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model.model(
                input_ids=ids, attention_mask=mask, position_ids=positions, past_key_values=past, use_cache=True
            )
        past = result.past_key_values
        logits = torch.nn.functional.linear(result.last_hidden_state[:, -1].float(), model.lm_head.weight)
        action = torch.multinomial(logits.softmax(-1), 1, generator=rng).squeeze(1)
        for i, (token, on) in enumerate(zip(action.tolist(), alive.tolist(), strict=True)):
            if on:
                suffix[i].append(token)
                stopped[i] = token == eos
        alive &= action != eos
        if not alive.any():
            break
        ids = torch.where(alive, action, torch.full_like(action, eos))[:, None]
        mask = torch.cat([mask, torch.ones_like(ids)], -1)
        positions = mask.sum(-1, keepdim=True) - 1
        if (step + 1) % 1024 == 0:
            print("CONTINUE", a.arm, step + 1, flush=True)
    combined = []
    for i, row in enumerate(rows):
        response = row["response_ids"] + suffix[i]
        combined.append(
            dict(
                row,
                original_grade=row["grade"],
                original_response_length=len(row["response_ids"]),
                original_cap=row["cap"],
                cap=row["cap"] + protocol["additional_token_cap"],
                original_finish_reason=row["finish_reason"],
                response_ids=response,
                suffix_ids=suffix[i],
                output=tokenizer.decode(response, skip_special_tokens=False),
                finish_reason="stop" if stopped[i] else "length",
                additional_token_cap=protocol["additional_token_cap"],
                continuation_seed=selected["seed"],
            )
        )
    fc.grade_math(combined, a.repo)
    dump(
        target,
        dict(
            arm=a.arm,
            protocol=protocol,
            rows=combined,
            complete=True,
            selected_cases=len(rows),
            recovered_correct=sum(r["grade"]["acc"] for r in combined),
        ),
    )
    print("CONTINUATION DONE", a.arm, sum(r["grade"]["acc"] for r in combined), "/", len(rows), flush=True)


def main():
    ap = argparse.ArgumentParser()
    for key in ["models", "out", "cache", "repo"]:
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--arm", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.set_per_process_memory_fraction(0.8)
    run(a)


if __name__ == "__main__":
    main()
