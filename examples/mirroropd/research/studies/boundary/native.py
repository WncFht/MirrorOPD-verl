# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Buffer-preserving native generation, teacher caching and TB0 numerical audits."""

import argparse
import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.mirroropd.assets.paths import resolve_location
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.core.metrics import fixed_metrics
from examples.mirroropd.research.studies.boundary.common import CACHE, GROUPS, MODELS, OUT, model_path


def setup():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load(name, student=False, parallel=False, path=None):
    mapping = {"": "cuda:0"}
    if parallel:
        mapping = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 1, "lm_head": 1}
        mapping.update({f"model.layers.{i}": int(i >= 14) for i in range(28)})
    model = AutoModelForCausalLM.from_pretrained(
        resolve_location(path) if path is not None else model_path(name),
        dtype=torch.float32 if student else torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=mapping,
        local_files_only=True,
    ).eval()
    if not student:
        model.requires_grad_(False)
    for key, buffer in model.named_buffers():
        if "inv_freq" in key:
            assert buffer.dtype == torch.float32, (key, buffer.dtype)
    return model


def hidden(model, row):
    ids = row["prompt_ids"] + row["response_ids"][:-1]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        value = model.model(
            input_ids=torch.tensor([ids], device=model.model.embed_tokens.weight.device), use_cache=False
        )
    return value.last_hidden_state[0, len(row["prompt_ids"]) - 1 :]


def bank(student):
    rows = []
    for shard in range(4):
        rows.extend(read(OUT / "banks" / student / f"shard-{shard}.json"))
    rows.sort(key=lambda r: r["id"])
    assert len(rows) == 640 and len({r["id"] for r in rows}) == 640
    return rows


@torch.no_grad()
def generate(name, shard):
    destination = OUT / "banks" / name / f"shard-{shard}.json"
    if destination.exists():
        assert len(read(destination)) == 160
        return
    rows = read(OUT / "prompts.json")[160 * shard : 160 * (shard + 1)]
    model = load(name)
    tokenizer = AutoTokenizer.from_pretrained(model_path("r1"), local_files_only=True)
    eos = tokenizer.eos_token_id
    head = model.lm_head.weight.float()
    output = []
    for start in range(0, len(rows), 16):
        batch = rows[start : start + 16]
        width = max(len(r["prompt_ids"]) for r in batch)
        ids = torch.tensor([[eos] * (width - len(r["prompt_ids"])) + r["prompt_ids"] for r in batch], device="cuda")
        mask = torch.tensor(
            [[0] * (width - len(r["prompt_ids"])) + [1] * len(r["prompt_ids"]) for r in batch], device="cuda"
        )
        positions = mask.cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 0)
        seeds = [52000000 + 100000 * int(name == "r7") + 10 * r["id"] for r in batch]
        rngs = [torch.Generator(device="cuda").manual_seed(s) for s in seeds]
        active = torch.ones(len(batch), device="cuda", dtype=torch.bool)
        sequences = [[] for _ in batch]
        past = None
        for step in range(1024):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model.model(
                    input_ids=ids, attention_mask=mask, position_ids=positions, past_key_values=past, use_cache=True
                )
            past = result.past_key_values
            logits = torch.nn.functional.linear(result.last_hidden_state[:, -1].float(), head)
            probabilities = logits.softmax(-1)
            actions = torch.stack(
                [torch.multinomial(p, 1, generator=g)[0] for p, g in zip(probabilities, rngs, strict=True)]
            )
            for i, (token, on) in enumerate(zip(actions.tolist(), active.tolist(), strict=True)):
                if on:
                    sequences[i].append(token)
            active &= actions != eos
            if not active.any():
                break
            ids = torch.where(active, actions, torch.full_like(actions, eos))[:, None]
            mask = torch.cat([mask, torch.ones_like(ids)], -1)
            positions = mask.sum(-1, keepdim=True) - 1
        for row, sequence, seed in zip(batch, sequences, seeds, strict=True):
            output.append(
                dict(
                    row,
                    response_ids=sequence,
                    source=f"tb-s0-{name}",
                    seed=seed,
                    cap=1024,
                    finish_reason="stop" if sequence[-1] == eos else "length",
                )
            )
        dump(OUT / "banks" / name / f"progress-{shard}.json", dict(generated=len(output), total=160))
        print("BANK", name, shard, len(output), flush=True)
        del past, result
    dump(destination, output)


@torch.no_grad()
def cache(teacher, student):
    root = CACHE / "teachers" / teacher / student
    if (root / "complete.json").exists():
        return
    model = load(teacher)
    rows = [r for r in bank(student) if r["group"] in GROUPS]
    root.mkdir(parents=True, exist_ok=True)
    torch.save(model.lm_head.weight.detach().cpu(), root / "head.pt")
    for i, row in enumerate(rows, 1):
        path = root / f"{row['id']:04d}.pt"
        if not path.exists():
            tensor = hidden(model, row).cpu()
            torch.save(
                dict(
                    prompt_ids=row["prompt_ids"],
                    response_ids=row["response_ids"],
                    hidden=tensor,
                    teacher=MODELS[teacher],
                    source=row["source"],
                ),
                path,
            )
        if i % 32 == 0:
            print("CACHE", teacher, student, i, len(rows), flush=True)
    dump(root / "complete.json", dict(rows=len(rows), teacher=MODELS[teacher], student=MODELS[student]))


def audit(name):
    destination = OUT / "numerical" / f"{name}.json"
    if destination.exists() and (name not in ["r1", "r7"] or (OUT / "numerical" / f"self-{name}.json").exists()):
        return
    model = load(name)
    head = model.lm_head.weight.float()
    rows = [bank("r1")[0], bank("r1")[320]]
    probes = []
    with torch.no_grad():
        for row in rows:
            ids = row["prompt_ids"] + row["response_ids"][:-1]
            selected = [
                len(row["prompt_ids"]) - 1,
                len(row["prompt_ids"]) + 127,
                len(row["prompt_ids"]) + 511,
                len(ids) - 1,
            ]
            selected = sorted({i for i in selected if i < len(ids)})
            positions = torch.tensor(selected, device="cuda")
            # The model's own CausalLM.forward with a FP32 head is the native
            # reference for the split hidden/head scorer. Buffers stay untouched.
            model.lm_head.float()
            hook = model.lm_head.register_forward_pre_hook(lambda module, args: (args[0].float(),))
            reference = (
                model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=positions)
                .logits[0]
                .float()
            )
            hook.remove()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False).last_hidden_state[
                    0, positions
                ]
            scores = torch.nn.functional.linear(h.float(), head)
            distances = fixed_metrics(scores, reference)
            probes.append(
                dict(
                    id=row["id"],
                    positions=selected,
                    max_logit_difference=float((scores - reference).abs().max()),
                    max_rkl=float(distances["rkl"].abs().max()),
                )
            )
    assert max(x["max_rkl"] for x in probes) < 1e-5, probes
    dump(
        destination,
        dict(
            teacher=MODELS[name],
            probes=probes,
            buffer_dtypes={k: str(v.dtype) for k, v in model.named_buffers()},
            reference="native CausalLM with FP32 head; BF16 backbone",
            time=time.time(),
        ),
    )
    del model, head
    gc.collect()
    torch.cuda.empty_cache()
    if name in ["r1", "r7"]:
        student_model = load(name, student=True)
        tiny = dict(rows[0], response_ids=rows[0]["response_ids"][:8])
        hp = hidden(student_model, tiny).float()
        logits = torch.nn.functional.linear(hp, student_model.lm_head.weight)
        lp = logits.log_softmax(-1)
        loss = (lp.exp() * (lp - lp.detach())).sum() / (16 * 1024)
        loss.backward()
        norm2 = sum(
            float(p.grad.square().sum(dtype=torch.float64)) for p in student_model.parameters() if p.grad is not None
        )
        assert abs(float(loss)) < 1e-8 and norm2**0.5 < 1e-5
        dump(
            OUT / "numerical" / f"self-{name}.json",
            dict(
                loss=float(loss),
                grad_norm=norm2**0.5,
                response_positions=8,
                precision="FP32 master/head, BF16 backbone",
            ),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["generate", "cache", "audit"])
    parser.add_argument("name", choices=MODELS)
    parser.add_argument("--student", choices=["r1", "r7"], default="r1")
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    setup()
    if args.action == "generate":
        generate(args.name, args.shard)
    elif args.action == "cache":
        cache(args.name, args.student)
    else:
        audit(args.name)


if __name__ == "__main__":
    main()
