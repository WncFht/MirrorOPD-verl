# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Frozen question identities, family-specific prompts and reusable S0 banks."""

import re
from dataclasses import asdict
from functools import lru_cache

from examples.mirroropd.research.backends.hf_models import load
from examples.mirroropd.research.core.artifacts import complete, file_hash, receipt
from examples.mirroropd.research.core.io import dump as write
from examples.mirroropd.research.core.io import read
from examples.mirroropd.research.studies.fixed_transfer.spec import fingerprint, resolve

PRIMARY_GROUPS = tuple("ABCDEFGH") + ("M01", "M02", "M03", "M04", "Code01", "Code02", "IF01", "IF02")


def normalized(value):
    return re.sub(r"\s+", " ", value).strip()


def question_messages(row):
    if "messages" in row:
        return row["messages"]
    # The imported historical source is explicitly an R1 single-user envelope.
    prefix = "<｜begin▁of▁sentence｜><｜User｜>"
    marker = "<｜Assistant｜>"
    raw = row["prompt"]
    if not raw.startswith(prefix) or raw.count(marker) != 1 or raw.count("<｜User｜>") != 1:
        raise ValueError(f"Unrecognized historical prompt envelope for {row['id']}")
    question, suffix = raw[len(prefix) :].split(marker)
    if suffix != "<think>\n":
        raise ValueError(f"Unexpected assistant suffix for {row['id']}")
    return [dict(role="user", content=question)]


def apply_prompt_policy(messages, policy):
    if not policy or "system_prompt" not in policy:
        return messages
    if any(m["role"] == "system" for m in messages):
        raise ValueError("Do not silently replace an existing system instruction")
    return [dict(role="system", content=policy["system_prompt"]), *messages]


def user_content(messages):
    return normalized("\n".join(m["content"] for m in messages if m["role"] == "user"))


def prompt_tokenizers(study):
    """Keep student tokenization; an explicit pinned template is a separate experimental condition."""
    from transformers import AutoTokenizer

    student = AutoTokenizer.from_pretrained(study.models[study.student].directory, local_files_only=True)
    source = (study.prompt_policy or {}).get("template_model")
    template = None
    if source is not None:
        template = AutoTokenizer.from_pretrained(study.models[source].directory, local_files_only=True)
        if student.get_vocab() != template.get_vocab():
            raise ValueError("Template controls require identical token-to-ID maps")
        if not isinstance(template.chat_template, str) or not template.chat_template:
            raise ValueError("Template source requires one explicit chat template")
        student.chat_template = template.chat_template
    return student, template


def prepare(study):
    destination = study.directory / "prepared"
    inputs = dict(
        student=study.models[study.student].identity,
        question_source=file_hash(resolve(study.question_source)),
        benchmark_source=file_hash(resolve(study.benchmark_source)),
        evaluation=asdict(study.evaluation),
        generation=dict(horizon=study.training.horizon, seed=study.training.bank_seed),
    )
    if study.prompt_policy is not None:
        inputs["prompt_policy"] = study.prompt_policy
    tokenizer, template = prompt_tokenizers(study)
    if template is not None:
        source = study.prompt_policy["template_model"]
        inputs["template_source"] = dict(
            model=study.models[source].identity,
            template=fingerprint(template.chat_template),
        )
    if complete(destination, inputs):
        return read(destination / "complete.json")

    def render(messages):
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=study.evaluation.thinking,
            return_dict=False,
            **(study.prompt_policy or {}).get("template_kwargs", {}),
        )
        if template is not None:
            native_ids = template.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                enable_thinking=study.evaluation.thinking,
                **(study.prompt_policy or {}).get("template_kwargs", {}),
            )
            if ids != native_ids:
                raise ValueError("Student tokenization differs from the declared native template control")
        if len(ids) > study.evaluation.model_length - study.evaluation.response_cap:
            raise ValueError("Prompt does not fit the shared evaluation context budget")
        return ids

    questions = []
    for row in read(resolve(study.question_source)):
        messages = apply_prompt_policy(question_messages(row), study.prompt_policy)
        ids = render(messages)
        questions.append(
            dict(
                row,
                messages=messages,
                prompt_ids=ids,
                prompt=tokenizer.decode(ids, skip_special_tokens=False),
                question_source=row["source"],
            )
        )
    benchmarks = []
    for row in read(resolve(study.benchmark_source)):
        messages = apply_prompt_policy(row["messages"], study.prompt_policy)
        ids = render(messages)
        benchmarks.append(
            dict(row, messages=messages, prompt_ids=ids, input=tokenizer.decode(ids, skip_special_tokens=False))
        )
    train = {user_content(r["messages"]) for r in questions if r["split"] == "train"}
    evaluation = {user_content(r["messages"]) for r in benchmarks}
    if train & evaluation:
        raise ValueError("Training questions overlap the task evaluation set")
    if len(questions) != 640 or len({r["id"] for r in questions}) != 640 or len(benchmarks) != 630:
        raise ValueError("The paired study requires the frozen 640/630 question cohorts")
    write(destination / "bank-prompts.json", questions)
    write(destination / "evaluation-prompts.json", benchmarks)
    bank_inputs = dict(
        student=inputs["student"],
        prompts=fingerprint(questions),
        horizon=study.training.horizon,
        seed=study.training.bank_seed,
        temperature=1.0,
        stop_ids=study.evaluation.stop_ids,
        precision="BF16 native backbone; FP32 head; per-prompt CUDA RNG",
    )
    receipt(
        destination,
        inputs,
        bank_key=fingerprint(bank_inputs),
        bank_inputs=bank_inputs,
        evaluation_key=fingerprint(benchmarks),
        train_eval_overlap=0,
    )
    return read(destination / "complete.json")


def bank_directory(study):
    return study.cache_directory / "banks" / read(study.directory / "prepared/complete.json")["bank_key"]


@lru_cache(maxsize=4)
def _read_bank(root, batch_size):
    """A sealed S0 bank is immutable; cache parsed tokens within each worker."""
    rows = []
    count = 640 // batch_size
    for index in range(count):
        rows.extend(read(root / f"batch-{index:03d}.json"))
    rows.sort(key=lambda r: r["id"])
    assert len(rows) == 640 and len({r["id"] for r in rows}) == 640
    assert all(r["source"] == root.name for r in rows)
    return tuple(rows)


def bank_rows(study):
    return list(_read_bank(bank_directory(study), study.training.batch_size))


def score_rows(study):
    return [r for r in bank_rows(study) if r["group"] in PRIMARY_GROUPS]


def required_rows(study, arm):
    count = (
        study.continuation.get("pool_batches", 20)
        if study.continuation is not None
        else arm.updates // study.training.reuse
    )
    groups = {chr(65 + i) for i in range(count)} | set(PRIMARY_GROUPS)
    return [r for r in bank_rows(study) if r["group"] in groups]


def bank_shard_receipt(study, shard, shards=4, *, groups=None):
    manifest = read(study.directory / "prepared/complete.json")
    root = bank_directory(study)
    inputs = dict(bank_key=manifest["bank_key"], shard=shard, shards=shards)
    directory = root / f"shard-{shard}"
    if groups is not None:
        groups = tuple(sorted(set(groups)))
        if not groups:
            raise ValueError("A selected bank panel must not be empty")
        inputs["groups"] = groups
        directory = root / f"panel-{fingerprint(groups)[:12]}-shard-{shard}"
    return directory, inputs


def generate(study, shard=0, shards=4, *, groups=None):
    import torch

    if not 0 <= shard < shards:
        raise ValueError("Invalid bank shard")
    root = bank_directory(study)
    manifest = read(study.directory / "prepared/complete.json")
    receipt_directory, inputs = bank_shard_receipt(study, shard, shards, groups=groups)
    if complete(receipt_directory, inputs):
        return
    prompts = read(study.directory / "prepared/bank-prompts.json")
    if groups is not None and not set(groups) <= {r["group"] for r in prompts}:
        raise ValueError("Unknown bank panel groups")
    model = load(study.models[study.student])
    head = model.get_output_embeddings().weight.detach().float()
    eos = study.evaluation.stop_ids[0]
    stops = torch.tensor(study.evaluation.stop_ids, device="cuda")
    batch_size = study.training.batch_size
    with torch.no_grad():
        for index in range(shard, len(prompts) // batch_size, shards):
            batch = prompts[index * batch_size : (index + 1) * batch_size]
            if groups is not None:
                selected = [r for r in batch if r["group"] in groups]
                if not selected:
                    continue
                if len(selected) != len(batch):
                    raise ValueError("Selected panels must preserve original generation batches")
            path = root / f"batch-{index:03d}.json"
            if path.exists():
                old = read(path)
                assert len(old) == batch_size and all(r["source"] == manifest["bank_key"] for r in old)
                continue
            width = max(len(r["prompt_ids"]) for r in batch)
            ids = torch.tensor([[eos] * (width - len(r["prompt_ids"])) + r["prompt_ids"] for r in batch], device="cuda")
            mask = torch.tensor(
                [[0] * (width - len(r["prompt_ids"])) + [1] * len(r["prompt_ids"]) for r in batch], device="cuda"
            )
            positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
            seeds = [study.training.bank_seed + 10 * r["id"] for r in batch]
            rngs = [torch.Generator(device="cuda").manual_seed(seed) for seed in seeds]
            active = torch.ones(len(batch), device="cuda", dtype=torch.bool)
            tokens = []
            past = None
            for step in range(study.training.horizon):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    value = model.get_decoder()(
                        input_ids=ids, attention_mask=mask, position_ids=positions, past_key_values=past, use_cache=True
                    )
                past = value.past_key_values
                probabilities = torch.nn.functional.linear(value.last_hidden_state[:, -1].float(), head).softmax(-1)
                actions = torch.stack(
                    [torch.multinomial(p, 1, generator=g)[0] for p, g in zip(probabilities, rngs, strict=True)]
                )
                tokens.append(torch.where(active, actions, eos))
                active &= ~torch.isin(actions, stops)
                if step % 32 == 31 and not bool(active.any()):
                    break
                ids = torch.where(active, actions, eos)[:, None]
                mask = torch.cat([mask, torch.ones_like(ids)], -1)
                positions = mask.sum(-1, keepdim=True) - 1
            sequences = torch.stack(tokens, dim=1).tolist()
            output = []
            for row, sequence, seed in zip(batch, sequences, seeds, strict=True):
                stop = next((i + 1 for i, t in enumerate(sequence) if t in study.evaluation.stop_ids), len(sequence))
                sequence = sequence[:stop]
                output.append(
                    dict(
                        row,
                        response_ids=sequence,
                        seed=seed,
                        source=manifest["bank_key"],
                        cap=study.training.horizon,
                        finish_reason="stop" if sequence[-1] in study.evaluation.stop_ids else "length",
                    )
                )
            write(path, output)
            print("BANK", study.name, shard, index, flush=True)
            del past, value, tokens
    receipt(receipt_directory, inputs)
