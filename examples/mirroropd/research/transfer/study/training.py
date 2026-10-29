# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared teacher caches and continuous Adam trajectories with weight-only readouts."""

import gc
import importlib.metadata
import math
import random
import time
from dataclasses import asdict

import torch

from . import engine, models
from .artifacts import complete, file_hash, read, receipt, write
from .data import bank_rows, required_rows
from .spec import Execution, fingerprint


def arm_by_name(study, name):
    return next(a for a in study.arms if a.name == name)


def teacher_inputs(study, name):
    rows = bank_rows(study)
    return dict(
        teacher=study.models[name].identity,
        bank=fingerprint(rows),
        execution="native BF16 backbone; FP32 head; native support",
        backend={name: importlib.metadata.version(name) for name in ["torch", "transformers", "flash-attn"]},
        contract=1,
    )


def teacher_directory(study, name):
    return study.cache_directory / "teachers" / fingerprint(teacher_inputs(study, name))


@torch.no_grad()
def cache_teacher(study, name):
    inputs = teacher_inputs(study, name)
    root = teacher_directory(study, name)
    relevant = [a for a in study.arms if a.teacher == name]
    chosen = {r["id"]: r for a in relevant for r in required_rows(study, a)}
    rows = [chosen[i] for i in sorted(chosen)]
    if complete(root, inputs) and all((root / f"{r['id']:04d}.pt").exists() for r in rows):
        return
    model = models.load(study.models[name])
    head = model.get_output_embeddings().weight.detach().float()
    probes = []
    for row in [rows[0], next(r for r in rows if r["group"] == "M01")]:
        ids = row["prompt_ids"] + row["response_ids"][:-1]
        relative = sorted({0, min(127, len(row["response_ids"]) - 1), len(row["response_ids"]) - 1})
        absolute = torch.tensor([len(row["prompt_ids"]) - 1 + i for i in relative], device="cuda")

        def head_hook(module, args, output):
            with torch.autocast("cuda", enabled=False):
                return torch.nn.functional.linear(args[0].float(), head)

        hook = model.get_output_embeddings().register_forward_hook(head_hook)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reference = (
                model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=absolute)
                .logits[0]
                .float()
            )
        hook.remove()
        hidden = models.hidden(model, [row], pad_id=study.evaluation.stop_ids[0])[relative].float()
        actual = torch.nn.functional.linear(hidden, head)
        difference = float((actual - reference).abs().max())
        probes.append(dict(id=row["id"], max_logit_difference=difference))
        assert difference < 1e-4, probes[-1]
    engine.save_tensor(root / "head.pt", model.get_output_embeddings().weight.detach().cpu())
    for i, row in enumerate(rows):
        path = root / f"{row['id']:04d}.pt"
        if not path.exists():
            value = models.hidden(model, [row], pad_id=study.evaluation.stop_ids[0]).cpu()
            engine.save_tensor(
                path,
                dict(
                    prompt_ids=row["prompt_ids"],
                    response_ids=row["response_ids"],
                    source=row["source"],
                    teacher=study.models[name].identity,
                    hidden=value,
                ),
            )
        if i % 64 == 0:
            print("TEACHER_CACHE", name, i, len(rows), flush=True)
    receipt(
        root,
        inputs,
        rows=len(rows),
        native_forward_probes=probes,
        buffer_dtypes={k: str(v.dtype) for k, v in model.named_buffers()},
    )


def teacher_state(study, name, rows):
    root = teacher_directory(study, name)
    if not complete(root, teacher_inputs(study, name)):
        raise ValueError(f"Teacher cache incomplete: {name}")
    head = torch.load(root / "head.pt", map_location="cuda", weights_only=True).float()
    states = {}
    for row in rows:
        value = torch.load(root / f"{row['id']:04d}.pt", map_location="cpu", weights_only=True)
        assert value["teacher"] == study.models[name].identity and value["source"] == row["source"]
        assert value["prompt_ids"] == row["prompt_ids"] and value["response_ids"] == row["response_ids"]
        states[row["id"]] = value["hidden"]
    return head, states


def arm_directory(study, arm):
    return study.directory / "arms" / arm.name


def checkpoint_directory(study, arm, step):
    return study.cache_directory / "checkpoints" / study.name / arm.name / f"step{step:03d}"


def scientific_inputs(study, arm):
    return dict(
        student=study.models[study.student].identity,
        teacher=study.models[arm.teacher].identity,
        bank=teacher_inputs(study, arm.teacher)["bank"],
        training=asdict(study.training),
        arm=asdict(arm),
    )


def export_weights(study, arm, step, model, inputs):
    from transformers import AutoTokenizer

    path = checkpoint_directory(study, arm, step)
    path.mkdir(parents=True, exist_ok=True)
    # Let HF remove tied aliases before serializing; copying each state_dict entry would duplicate them.
    model.save_pretrained(path, max_shard_size="4GB", safe_serialization=True)
    AutoTokenizer.from_pretrained(study.models[study.student].directory, local_files_only=True).save_pretrained(path)
    files = [dict(name=p.name, sha256=file_hash(p), bytes=p.stat().st_size) for p in sorted(path.glob("*.safetensors"))]
    content = fingerprint(dict(weights=files, config=file_hash(path / "config.json")))
    receipt(
        path,
        dict(inputs, step=step),
        weight_identity=content,
        files=files,
        dtype="float32",
        optimizer_saved=False,
        transient=step != arm.updates,
    )
    return content


def train(study, name):
    arm = arm_by_name(study, name)
    directory = arm_directory(study, arm)
    from .validation import profile_directory

    near = next(a.teacher for a in study.arms if study.models[a.teacher].method == "rl")
    validated = read(profile_directory(study, near) / "complete.json")
    execution = Execution(**validated["execution"])
    inputs = dict(
        scientific_inputs(study, arm), execution=asdict(execution), numerical_profile=validated["fingerprint"]
    )
    if complete(directory, inputs):
        return
    if (directory / "training.json").exists():
        raise RuntimeError("Partial scientific run exists; preserve it and use a distinct attempt")
    rows = required_rows(study, arm)
    assert arm.updates // study.training.reuse <= len({r["group"] for r in rows if r["split"] == "train"})
    random.seed(study.training.seed)
    torch.manual_seed(study.training.seed)
    head, states = teacher_state(study, arm.teacher, rows)
    model = models.load(study.models[study.student], trainable=True)
    models.training_mode(model, execution)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=study.training.learning_rate,
        betas=study.training.betas,
        eps=study.training.epsilon,
        weight_decay=study.training.weight_decay,
        foreach=False,
    )
    write(
        directory / "protocol.json",
        dict(
            inputs=inputs,
            execution=asdict(execution),
            parameter_dtype="float32",
            optimizer_dtype="float32",
            rollout_source="fixed S0",
            save_policy="final weights; transient readout weights; no Adam",
        ),
    )
    pad = study.evaluation.stop_ids[0]
    write(directory / "fixed-000.json", engine.score(model, head, states, rows, 0, study.training, execution, pad))
    history = []
    qcache = None
    score_steps = study.training.score_steps(arm.updates, arm.evaluate_at)
    started = time.time()
    for step in range(1, arm.updates + 1):
        group = chr(65 + (step - 1) // study.training.reuse)
        batch = [r for r in rows if r["group"] == group]
        assert len(batch) == study.training.batch_size and all(r["split"] == "train" for r in batch)
        cache_seconds = 0.0
        if qcache is None or (step - 1) % study.training.reuse == 0:
            del qcache
            begin = time.perf_counter()
            qcache = engine.log_probabilities(head, states, batch, execution.token_chunk)
            torch.cuda.synchronize()
            cache_seconds = time.perf_counter() - begin
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        loss = engine.backward(model, qcache, batch, study.training, execution, pad)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), study.training.clip, foreach=False)
        assert math.isfinite(loss) and bool(torch.isfinite(norm))
        torch.cuda.synchronize()
        backward_done = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        record = dict(
            step=step,
            group=group,
            loss=loss,
            grad_norm=float(norm),
            cache_seconds=cache_seconds,
            backward_seconds=backward_done - begin,
            optimizer_seconds=time.perf_counter() - backward_done,
            update_seconds=time.perf_counter() - begin,
            actual_tokens=sum(len(r["response_ids"]) for r in batch),
            peak_gib=torch.cuda.max_memory_allocated() / 2**30,
            elapsed=time.time() - started,
        )
        history.append(record)
        write(directory / "training.json", history)
        print("UPDATE", name, record, flush=True)
        optimizer.zero_grad(set_to_none=True)
        if step in score_steps:
            del qcache
            qcache = None
            write(
                directory / f"fixed-{step:03d}.json",
                engine.score(model, head, states, rows, step, study.training, execution, pad),
            )
        if step in arm.evaluate_at:
            assert {int(s["step"]) for s in optimizer.state.values()} == {step}
            export_weights(study, arm, step, model, inputs)
    receipt(directory, inputs, updates=arm.updates, elapsed=time.time() - started)
    del model, optimizer, head, states
    gc.collect()


def evaluate_endpoint(study, name, step):
    from .evaluation import evaluate, evaluation_inputs

    arm = arm_by_name(study, name)
    path = checkpoint_directory(study, arm, step)
    saved = read(path / "complete.json")
    expected = evaluation_inputs(study, study.student, saved["weight_identity"])
    evaluation_directory = study.cache_directory / "evaluations" / fingerprint(expected)
    if not complete(evaluation_directory, expected):
        if saved.get("weights_pruned"):
            raise RuntimeError("This transient readout was pruned after evaluation; regenerate it for new evaluation")
        for item in saved["files"]:
            source = path / item["name"]
            if source.stat().st_size != item["bytes"] or file_hash(source) != item["sha256"]:
                raise ValueError(f"Checkpoint content changed: {source}")
    evaluate(study, study.student, weights=path, weight_identity=saved["weight_identity"], label=f"{name}-{step:03d}")


def training_jobs(study, existing):
    from .validation import copy_inputs, profile_directory, profile_inputs

    result = []
    banks = [f"bank-{i}" for i in range(4)]
    for name in sorted({a.teacher for a in study.arms}):
        result.append(
            dict(
                name="cache-" + name,
                args=["cache", "--model", name],
                dependencies=banks,
                marker=str(teacher_directory(study, name) / "complete.json"),
                fingerprint=fingerprint(teacher_inputs(study, name)),
                retries=1,
            )
        )
    near = next(a.teacher for a in study.arms if study.models[a.teacher].method == "rl")
    result.append(
        dict(
            name="numerical",
            args=["validate", "--model", near],
            dependencies=["cache-" + near],
            marker=str(profile_directory(study, near) / "complete.json"),
            fingerprint=fingerprint(profile_inputs(study, near)),
            retries=0,
        )
    )
    for arm in study.arms:
        result.append(
            dict(
                name="train-" + arm.name,
                args=["train", "--arm", arm.name],
                dependencies=[
                    "cache-" + arm.teacher,
                    "numerical",
                    "baseline-" + study.student,
                    "baseline-" + arm.teacher,
                ],
                marker=str(arm_directory(study, arm) / "complete.json"),
                expected_inputs=scientific_inputs(study, arm),
                retries=0,
            )
        )
        for step in arm.evaluate_at:
            result.append(
                dict(
                    name=f"eval-{arm.name}-{step:03d}",
                    args=["endpoint", "--arm", arm.name, "--step", str(step)],
                    dependencies=[],
                    ready_file=str(checkpoint_directory(study, arm, step) / "complete.json"),
                    ready_inputs=dict(scientific_inputs(study, arm), step=step),
                    producer="train-" + arm.name,
                    completion_weight_from_ready=True,
                    marker=str(study.directory / "evaluations" / f"{arm.name}-{step:03d}/complete.json"),
                    retries=2,
                )
            )
    interfaces = read(study.directory / "interfaces.json")
    student = interfaces[study.student]["configuration"]
    for name in sorted({a.teacher for a in study.arms}):
        target = interfaces[name]["configuration"]
        if models.same_parameter_layout(student, target):
            result.append(
                dict(
                    name="copy-" + name,
                    args=["copy", "--model", name],
                    dependencies=["cache-" + name],
                    marker=str(study.directory / "teacher-copy" / name / "complete.json"),
                    fingerprint=fingerprint(copy_inputs(study, name)),
                    retries=0,
                )
            )
    priorities = {"cache": 20, "validate": 30, "train": 40, "copy": 45, "endpoint": 50}
    for job in result:
        job["priority"] = priorities[job["args"][0]]
    return result
