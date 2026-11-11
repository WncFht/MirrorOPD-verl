# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""GPU stages; generation, scoring and training run in separate processes."""

import json
import math
import random
import time

from .data import contract, fingerprint, jsonl, read, records, sha, write
from .objectives import log_probabilities, merged_ce_backward, merged_statistics, validate_content


def sample_seed(question_id, index, seed):
    return int(fingerprint([question_id, index, seed])[:8], 16) % (2**31 - 1)


def sampled_path_probabilities(raw, logprobs, end_ids):
    """Sum the sampled body and marginalize both stop draws at the final prefix."""
    if not raw or raw[-1] not in end_ids or logprobs is None or len(logprobs) != len(raw):
        return None, None
    if not all(e in logprobs[-1] for e in end_ids):
        return None, None
    body = math.fsum(float(logprobs[k][token].logprob) for k, token in enumerate(raw[:-1]))
    values = [float(logprobs[-1][e].logprob) for e in end_ids]
    peak = max(values)
    if not math.isfinite(peak):
        return None, None
    end = peak + math.log(math.fsum(math.exp(v - peak) for v in values))
    return body, end


def grade_candidate(value):
    from examples.mirroropd.research.core.math_grading import grade
    from verl.trainer.evaluation.accuracy import final_answer
    from verl.utils.reward_score.math_dapo import compute_score

    result = grade(
        dict(data_source="DAPO", output=value["text"], gts=value["truth"], response_ids=value["content_ids"])
    )
    answer = final_answer(value["text"], thinking_open=False, channel="after_thinking")
    return dict(
        correct=float(result["acc"]) == 1,
        native_score=float(compute_score(answer, str(value["truth"]))["acc"]),
        repeat8=result["repeat8"],
    )


def generate(root, split, shard, shards):
    import concurrent.futures
    import multiprocessing

    from vllm import LLM, SamplingParams

    from examples.mirroropd.research.core.math_grading import init_grader

    cfg = contract(root)
    if shards != cfg["generation_shards"]:
        raise ValueError("Generation sharding differs from the frozen plan")
    questions = list(records(root / "prepared" / f"{split}.jsonl"))[shard::shards]
    quota = (cfg["counts"]["train"] + shards - 1 - shard) // shards if split == "train" else None
    covered = 0
    for q in questions:
        path = root / "generated" / split / f"{q['id']}.json"
        if path.exists():
            completed = read(path)
            if completed["generation_protocol"] != sha(root / "protocol.json"):
                raise ValueError("Generated candidates belong to a different protocol")
            covered += sum(c["correct"] and c["complete"] for c in completed["candidates"]) >= cfg["min_correct"]
    receipt = root / f"generation-{split}-{shard}-of-{shards}.json"
    if quota is not None and covered >= quota:
        write(receipt, dict(covered=covered, quota=quota, protocol_sha256=sha(root / "protocol.json")))
        return
    pending = [q for q in questions if not (root / "generated" / split / f"{q['id']}.json").exists()]
    if not pending:
        write(receipt, dict(covered=covered, quota=quota, protocol_sha256=sha(root / "protocol.json")))
        return
    llm = LLM(
        model=cfg["expert"],
        tensor_parallel_size=cfg["expert_tp"],
        dtype="bfloat16",
        max_model_len=cfg["max_length"],
        max_num_seqs=16,
        gpu_memory_utilization=0.88,
        enforce_eager=True,
        seed=cfg["generation_seed"],
        enable_prefix_caching=True,
        trust_remote_code=False,
        logprobs_mode="processed_logprobs",
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn"), initializer=init_grader
    ) as pool:
        for q in pending:
            partial = root / "partial" / split / f"{q['id']}.json"
            candidates = read(partial)["candidates"] if partial.exists() else []
            while len(candidates) < cfg["max_candidates"]:
                if (
                    len(candidates) >= cfg["initial_candidates"]
                    and sum(c["correct"] and c["complete"] for c in candidates) >= cfg["min_correct"]
                ):
                    break
                start = len(candidates)
                params = [
                    SamplingParams(
                        n=1,
                        temperature=1.0,
                        top_p=1.0,
                        top_k=-1,
                        repetition_penalty=1.0,
                        max_tokens=cfg["max_response"],
                        stop_token_ids=cfg["end_ids"],
                        allowed_token_ids=list(range(cfg["support_size"])),
                        ignore_eos=False,
                        seed=sample_seed(q["id"], j, cfg["generation_seed"]),
                        logprobs=2,
                        logprob_token_ids=cfg["end_ids"],
                    )
                    for j in range(start, start + 16)
                ]
                outputs = llm.generate([dict(prompt_token_ids=q["expert_prompt_ids"])] * 16, params, use_tqdm=False)
                batch, grades = [], []
                for j, result in enumerate(outputs, start):
                    output = result.outputs[0]
                    raw = list(output.token_ids)
                    terminal_in_ids = bool(raw and raw[-1] in cfg["end_ids"])
                    body = raw[:-1] if terminal_in_ids else raw
                    stop = raw[-1] if terminal_in_ids else output.stop_reason
                    complete = output.finish_reason == "stop" and stop in cfg["end_ids"]
                    try:
                        validate_content(
                            q["prompt_ids"],
                            body,
                            cfg["end_ids"],
                            max_prompt=cfg["max_prompt"],
                            max_response=cfg["max_response"],
                            max_length=cfg["max_length"],
                        )
                    except ValueError:
                        complete = False
                    logprobs = output.logprobs
                    sampled_lp, sampled_end = sampled_path_probabilities(raw, logprobs, cfg["end_ids"])
                    candidate = dict(
                        candidate_id=f"{q['id']}:{j}",
                        sample_index=j,
                        seed=params[j - start].seed,
                        text=llm.get_tokenizer().decode(body, skip_special_tokens=False),
                        content_ids=body,
                        raw_token_ids=raw,
                        finish_reason=output.finish_reason,
                        stop_reason=stop,
                        complete=complete,
                        sampled_content_logprob=sampled_lp,
                        sampled_end_logprob=sampled_end,
                    )
                    batch.append(candidate)
                    grades.append(pool.submit(grade_candidate, candidate | dict(truth=q["ground_truth"])))
                for candidate, future in zip(batch, grades, strict=True):
                    candidates.append(candidate | future.result())
                write(partial, dict(id=q["id"], candidates=candidates))
            write(
                root / "generated" / split / f"{q['id']}.json",
                dict(id=q["id"], candidates=candidates, generation_protocol=sha(root / "protocol.json")),
            )
            print(
                json.dumps(
                    dict(
                        stage="generated",
                        split=split,
                        id=q["id"],
                        candidates=len(candidates),
                        accepted=sum(c["complete"] and c["correct"] for c in candidates),
                    )
                ),
                flush=True,
            )
            covered += sum(c["complete"] and c["correct"] for c in candidates) >= cfg["min_correct"]
            if quota is not None and covered >= quota:
                break
    if quota is not None and covered < quota:
        raise ValueError("Training reservoir exhausted; do not recycle held-out questions")
    write(receipt, dict(covered=covered, quota=quota, protocol_sha256=sha(root / "protocol.json")))


def score(root, split, kind, shard, shards):
    import torch
    from transformers import AutoModelForCausalLM

    from examples.mirroropd.research.backends.hf_models import hidden

    cfg = contract(root)
    questions = list(records(root / "prepared" / f"{split}.jsonl"))[shard::shards]
    if split == "train":
        questions = [q for q in questions if (root / "generated" / split / f"{q['id']}.json").exists()]
    pending = [q for q in questions if not (root / "scores" / kind / split / f"{q['id']}.json").exists()]
    if not pending:
        return
    # Automatic layer placement uses only the scheduler's visible, reserved cards.
    model = None
    if kind == "base":
        from examples.mirroropd.research.studies.fixed_transfer.teacher_construction_llm import load_model

        model = load_model(cfg["base"], False)
    elif split == "calibration":
        model = (
            AutoModelForCausalLM.from_pretrained(
                cfg[kind],
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            .eval()
            .requires_grad_(False)
        )
        model.get_output_embeddings().float()
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        for q in pending:
            generated = root / "generated" / split / f"{q['id']}.json"
            pool = read(generated)
            values = {}
            for c in pool["candidates"]:
                if not (c["complete"] and c["correct"]):
                    continue
                prompt = q["prompt_ids"] if kind == "base" else q["expert_prompt_ids"]
                stats = {}
                if model is not None:
                    row = dict(prompt_ids=prompt, response_ids=c["content_ids"] + [cfg["end_ids"][0]])
                    states = hidden(model, [row], pad_id=cfg["end_ids"][0])
                    stats = merged_statistics(
                        states.to(model.get_output_embeddings().weight.device),
                        model.get_output_embeddings().weight,
                        c["content_ids"],
                        cfg["end_ids"],
                        chunk=cfg["token_chunk"],
                        support_size=cfg["support_size"],
                    )
                if kind == "expert":
                    source = c["sampled_content_logprob"]
                    end = c["sampled_end_logprob"]
                    if source is None or end is None or not math.isfinite(source + end):
                        raise ValueError("Missing complete-path sampler probabilities, including both END IDs")
                    if model is not None:
                        stats["hf_sequence_logprob"] = stats["sequence_logprob"]
                        stats["sampling_score_delta_per_token"] = (stats["content_logprob"] - source) / len(
                            c["content_ids"]
                        )
                    # Use the actual generation distribution, not a teacher-forced HF proxy.
                    stats.update(
                        content_logprob=source,
                        end_logprob=end,
                        sequence_logprob=source + end,
                        probability_source="vllm-processed-logprobs-with-merged-END",
                    )
                values[c["candidate_id"]] = stats
            write(
                root / "scores" / kind / split / f"{q['id']}.json",
                dict(id=q["id"], generated_sha256=sha(generated), candidates=values),
            )


def collect_scores(root, split):
    cfg = contract(root)
    for i in range(cfg["generation_shards"]):
        receipt = read(root / f"generation-{split}-{i}-of-{cfg['generation_shards']}.json")
        if receipt["protocol_sha256"] != sha(root / "protocol.json"):
            raise ValueError("Missing matching completed generation shard")
    deltas, lengths, counts, unique_counts = [], [], [], []
    disagreements = []
    correct = native_correct = 0
    truncated = total = 0
    for q in records(root / "prepared" / f"{split}.jsonl"):
        generation = root / "generated" / split / f"{q['id']}.json"
        if split == "train" and not generation.exists():
            continue
        raw = read(generation)
        scores = {name: read(root / "scores" / name / split / f"{q['id']}.json") for name in ("base", "expert")}
        if any(s["generated_sha256"] != sha(generation) for s in scores.values()):
            raise ValueError("Scoring belongs to a different candidate pool")
        accepted = []
        for c in raw["candidates"]:
            total += 1
            correct += c["correct"]
            native_correct += c["native_score"] == 1
            if c["correct"] != (c["native_score"] == 1) and len(disagreements) < 100:
                disagreements.append(
                    dict(
                        question_id=q["id"],
                        candidate_id=c["candidate_id"],
                        text=c["text"],
                        truth=q["ground_truth"],
                        canonical=c["correct"],
                        native=c["native_score"],
                    )
                )
            truncated += c["finish_reason"] == "length"
            lengths.append(len(c["content_ids"]))
            if c["complete"] and c["correct"]:
                scored = c | {name: s["candidates"][c["candidate_id"]] for name, s in scores.items()}
                delta = scored["expert"].get("sampling_score_delta_per_token")
                if split == "calibration" and (delta is None or not math.isfinite(delta)):
                    raise ValueError("Missing HF/sampler calibration comparison")
                if delta is not None:
                    deltas.append(abs(delta))
                accepted.append(scored)
        counts.append(len(accepted))
        unique_counts.append(len({tuple(c["content_ids"]) for c in accepted}))
        write(
            root / "scored" / split / f"{q['id']}.json",
            dict(id=q["id"], prompt_ids=q["prompt_ids"], ground_truth=q["ground_truth"], candidates=accepted),
        )
    lengths.sort()
    deltas.sort()
    if not sum(counts):
        raise ValueError("No correct complete candidates")
    report = dict(
        questions=len(counts),
        generated=total,
        complete_correct=sum(counts),
        canonical_accuracy=correct / max(total, 1),
        native_accuracy=native_correct / max(total, 1),
        unique_correct_candidates=sum(unique_counts),
        covered=sum(n >= cfg["min_correct"] for n in counts),
        truncation_fraction=truncated / max(total, 1),
        length_quantiles={str(p): lengths[min(len(lengths) - 1, int(p * (len(lengths) - 1)))] for p in (0.1, 0.5, 0.9)},
        sampling_delta_p95=deltas[int(0.95 * (len(deltas) - 1))] if deltas else None,
        sampling_delta_max=max(deltas) if deltas else None,
        precision=(
            "Expert denominator recorded from processed sampler probabilities, "
            "including merged END; HF calibration is diagnostic"
        ),
    )
    report["passed"] = (
        report["truncation_fraction"] <= cfg["max_truncation_fraction"]
        and report["covered"] / len(counts) >= cfg["min_calibration_coverage"]
        if split == "calibration"
        else report["covered"] >= cfg["counts"]["train"]
    )
    if split == "calibration":
        report["passed"] = report["passed"] and report["sampling_delta_max"] <= cfg["max_sampling_delta_per_token"]
    write(root / f"quality-{split}.json", report)
    write(root / f"grading-disagreements-{split}.json", disagreements)
    if not report["passed"]:
        raise ValueError(
            f"{split} pool failed quality checks; inspect quality-{split}.json and revise a fresh protocol"
        )


def load_training_inputs(root, seed):
    cfg = contract(root)
    if seed not in cfg["seeds"]:
        raise ValueError("Training seed is outside the frozen plan")
    frozen = read(root / "frozen.json")
    if frozen["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("Protocol changed after sampling freeze")
    for name, expected in frozen["files"].items():
        if sha(root / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    groups = {g["id"]: g for g in records(root / "pool.jsonl")}
    selection = list(records(root / f"selection-{seed}.jsonl"))
    return cfg, groups, selection


def train(root, arm, seed, resume=None):
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from examples.mirroropd.research.studies.fixed_transfer.teacher_construction_llm import (
        append_log,
        checkpoint,
        load_model,
        numerical_setup,
        train_mode,
    )

    cfg, groups, selection = load_training_inputs(root, seed)
    gate = read(root / "gpu-acceptance.json")
    if not gate["passed"] or gate["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("Missing matching long-context acceptance")
    out = root / "teachers" / f"{arm}-{seed}"
    if (out / "complete.json").exists():
        if read(out / "complete.json")["frozen_sha256"] != sha(root / "frozen.json"):
            raise ValueError("Completed teacher belongs to different frozen inputs")
        return
    if (out / "metrics.jsonl").exists() and resume is None:
        raise FileExistsError("Existing trajectory: specify an explicit complete checkpoint to resume")
    numerical_setup(seed)
    model = load_model(str(resume) if resume else cfg["base"], True)
    train_mode(model)
    tokenizer = AutoTokenizer.from_pretrained(cfg["base"], local_files_only=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=0.0, foreach=False)
    start, presented = 0, 0
    if resume:
        receipt = read(resume / "sft-state.json")
        if receipt["arm"] != arm or receipt["seed"] != seed or receipt["frozen_sha256"] != sha(root / "frozen.json"):
            raise ValueError("Checkpoint belongs to another frozen trajectory")
        state = torch.load(resume / "training-state.pt", weights_only=False, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        start, presented = receipt["step"], receipt["presented_tokens"]
        # Preserve the later failed attempt before continuing from a known optimizer state.
        metrics = out / "metrics.jsonl"
        if metrics.exists():
            history = list(records(metrics))
            if any(v["step"] > start for v in history):
                metrics.rename(out / f"metrics-before-resume-{time.time_ns()}.jsonl")
                jsonl(metrics, (v for v in history if v["step"] <= start))
    out.mkdir(parents=True, exist_ok=True)
    if len(selection) % cfg["batch"]:
        raise ValueError("Question presentations must divide the global batch")
    t0 = time.monotonic()
    lookup = {g["id"]: {c["candidate_id"]: c for c in g["candidates"]} for g in groups.values()}
    steps_per_epoch = len(groups) // cfg["batch"]
    pending_budgets = [b for b in read(root / "frozen.json").get("token_readout_budgets", []) if b > presented]
    for offset in range(start * cfg["batch"], len(selection), cfg["batch"]):
        step = offset // cfg["batch"] + 1
        optimizer.zero_grad(set_to_none=True)
        loss = 0.0
        batch_tokens = input_tokens = 0
        for draw in selection[offset : offset + cfg["batch"]]:
            group = groups[draw["question_id"]]
            candidate = lookup[group["id"]][draw["choices"][arm]]
            body = candidate["content_ids"]
            validate_content(
                group["prompt_ids"],
                body,
                cfg["end_ids"],
                max_prompt=cfg["max_prompt"],
                max_response=cfg["max_response"],
                max_length=cfg["max_length"],
            )
            loss += merged_ce_backward(
                model,
                dict(prompt_ids=group["prompt_ids"], content_ids=body),
                cfg["end_ids"],
                cfg["batch"] * cfg["reference_length"],
                chunk=cfg["token_chunk"],
                support_size=cfg["support_size"],
            )
            batch_tokens += len(body) + 1
            input_tokens += len(group["prompt_ids"]) + len(body)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"]))
        if not math.isfinite(loss) or not math.isfinite(norm):
            raise FloatingPointError("Nonfinite SFT update; retain the failed attempt")
        optimizer.step()
        presented += batch_tokens
        append_log(
            out / "metrics.jsonl",
            dict(
                step=step,
                loss=loss,
                gradient_norm=norm,
                clipped=norm > cfg["clip"],
                presented_tokens=presented,
                input_tokens=input_tokens,
                elapsed=time.monotonic() - t0,
                max_memory_allocated=torch.cuda.max_memory_allocated(),
            ),
        )
        crossed = [budget for budget in pending_budgets if presented >= budget]
        pending_budgets = [budget for budget in pending_budgets if budget > presented]
        if crossed or step in [steps_per_epoch * e for e in (1, 2, 4)]:
            dest = out / f"step{step:04d}"
            if dest.exists():
                raise FileExistsError("Checkpoint export cannot overwrite a prior attempt")
            checkpoint(model, tokenizer, optimizer, dest, step)
            write(
                dest / "sft-state.json",
                dict(
                    arm=arm,
                    seed=seed,
                    step=step,
                    presented_tokens=presented,
                    token_readout_targets=crossed,
                    token_budget_overshoot={str(b): presented - b for b in crossed},
                    frozen_sha256=sha(root / "frozen.json"),
                ),
            )
    write(
        out / "complete.json",
        dict(
            updates=len(selection) // cfg["batch"], presented_tokens=presented, frozen_sha256=sha(root / "frozen.json")
        ),
    )


def acceptance(root):
    """Real-model memory/update gate at 2K/8K/24K; never exports a scientific teacher."""
    import torch

    from examples.mirroropd.research.studies.fixed_transfer.teacher_construction_llm import (
        load_model,
        numerical_setup,
        train_mode,
    )

    cfg = contract(root)
    numerical_setup(cfg["seeds"][0])
    model = load_model(cfg["base"], True)
    train_mode(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=0.0, foreach=False)
    q = next(records(root / "prepared/calibration.jsonl"))
    prompt = (q["prompt_ids"] + [220] * cfg["max_prompt"])[: cfg["max_prompt"]]
    evidence = []
    for length in (2048, 8192, 24576):
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        body = ([220, 16, 17, 18, 13] * math.ceil((length - 1) / 5))[: length - 1]
        value = dict(prompt_ids=prompt, content_ids=body)
        validate_content(
            value["prompt_ids"],
            body,
            cfg["end_ids"],
            max_prompt=cfg["max_prompt"],
            max_response=cfg["max_response"],
            max_length=cfg["max_length"],
        )
        reference_grads = None
        reference_loss = None
        if length == 2048:
            from examples.mirroropd.research.backends.hf_models import hidden

            states = hidden(
                model, [dict(prompt_ids=prompt, response_ids=body + [cfg["end_ids"][0]])], pad_id=cfg["end_ids"][0]
            )
            lp = log_probabilities(states, model.get_output_embeddings().weight, cfg["support_size"])
            target = torch.tensor(body, device=lp.device)
            objective = (
                -(lp[:-1].gather(1, target[:, None]).sum() + torch.logsumexp(lp[-1, cfg["end_ids"]], 0))
                / cfg["reference_length"]
            )
            objective.backward()
            reference_loss = float(objective.detach())
            reference_grads = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
            del states, lp, objective
            optimizer.zero_grad(set_to_none=True)
        loss = merged_ce_backward(
            model,
            value,
            cfg["end_ids"],
            cfg["reference_length"],
            chunk=cfg["token_chunk"],
            support_size=cfg["support_size"],
        )
        comparison = None
        if reference_grads is not None:
            squared_error = squared_reference = 0.0
            for parameter, expected in zip(model.parameters(), reference_grads, strict=True):
                if (parameter.grad is None) != (expected is None):
                    raise ArithmeticError("Dense/chunked gradient support changed")
                if expected is not None:
                    squared_error += float((parameter.grad - expected).square().sum())
                    squared_reference += float(expected.square().sum())
            relative = math.sqrt(squared_error / max(squared_reference, 1e-30))
            if relative > 0.02 or abs(loss - reference_loss) > 1e-4 * max(1, abs(reference_loss)):
                raise ArithmeticError(f"Dense/chunked real-model mismatch: relative gradient error {relative}")
            comparison = dict(reference_loss=reference_loss, relative_gradient_l2=relative)
            del reference_grads, expected
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"]))
        weight = model.get_output_embeddings().weight
        before = weight[16].detach().clone()
        optimizer.step()
        delta = float((weight[16].detach() - before).abs().max())
        if not (math.isfinite(loss) and math.isfinite(norm) and norm > 0 and delta > 0):
            raise ArithmeticError("Long-context update failed")
        if any(
            p.dtype != torch.float32 or p.grad is not None and p.grad.dtype != torch.float32 for p in model.parameters()
        ):
            raise ArithmeticError("Master/gradient dtype changed")
        if any(
            s["exp_avg"].dtype != torch.float32 or s["exp_avg_sq"].dtype != torch.float32
            for s in optimizer.state.values()
        ):
            raise ArithmeticError("Optimizer moments changed dtype")
        evidence.append(
            dict(
                length=length,
                synthetic=True,
                prompt_tokens=len(prompt),
                dense_comparison=comparison,
                loss=loss,
                norm=norm,
                update_max=delta,
                memory=torch.cuda.max_memory_allocated(),
            )
        )
    write(
        root / "gpu-acceptance.json",
        dict(
            passed=True,
            protocol_sha256=sha(root / "protocol.json"),
            checks=evidence,
            tied_embeddings=model.get_input_embeddings().weight is model.get_output_embeddings().weight,
        ),
    )
