# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared teacher caches and continuous Adam trajectories with weight-only readouts."""

import gc
import importlib.metadata
import math
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch

from examples.mirroropd.research.backends import hf_models
from examples.mirroropd.research.backends import hf_rkl as engine
from examples.mirroropd.research.core import training_state
from examples.mirroropd.research.core.artifacts import complete, file_hash, receipt
from examples.mirroropd.research.core.io import dump as write
from examples.mirroropd.research.core.io import read
from examples.mirroropd.research.studies.fixed_transfer import models
from examples.mirroropd.research.studies.fixed_transfer.data import bank_rows, required_rows
from examples.mirroropd.research.studies.fixed_transfer.spec import Arm, Execution, fingerprint


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
        model_code=file_hash(hf_models.__file__),
        engine_code=file_hash(engine.__file__),
    )


def teacher_directory(study, name):
    return study.cache_directory / "teachers" / fingerprint(teacher_inputs(study, name))


@torch.no_grad()
def cache_teacher(study, name):
    inputs = teacher_inputs(study, name)
    root = teacher_directory(study, name)
    relevant = [a for a in study.arms if a.teacher == name]
    if not relevant:
        relevant = [reference_arm(study, name)]
    chosen = {r["id"]: r for a in relevant for r in required_rows(study, a)}
    rows = [chosen[i] for i in sorted(chosen)]
    if complete(root, inputs) and all((root / f"{r['id']:04d}.pt").exists() for r in rows):
        return
    model = hf_models.load(study.models[name])
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
        hidden = hf_models.hidden(model, [row], pad_id=study.evaluation.stop_ids[0])[relative].float()
        actual = torch.nn.functional.linear(hidden, head)
        difference = float((actual - reference).abs().max())
        probes.append(dict(id=row["id"], max_logit_difference=difference))
        assert difference < 1e-4, probes[-1]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(study.models[name].directory, local_files_only=True)
    mode_probes = []
    for row in [rows[0], next(r for r in rows if r["group"] == "M01")]:
        variants = {"shared_student": row["prompt_ids"]}
        if row.get("messages") and tokenizer.chat_template:
            native_ids = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                enable_thinking=study.evaluation.thinking,
                **(study.prompt_policy or {}).get("template_kwargs", {}),
            )
            if native_ids != row["prompt_ids"]:
                variants["native_teacher"] = native_ids
        for kind, ids in variants.items():
            hook = model.get_output_embeddings().register_forward_hook(head_hook)
            try:
                output = model.generate(
                    input_ids=torch.tensor([ids], device="cuda"),
                    max_new_tokens=32,
                    do_sample=False,
                    eos_token_id=list(study.evaluation.stop_ids),
                    pad_token_id=study.evaluation.stop_ids[0],
                    use_cache=True,
                )[0, len(ids) :].tolist()
            finally:
                hook.remove()
            mode_probes.append(
                dict(
                    id=row["id"],
                    prefix=kind,
                    prompt_ids=ids,
                    response_ids=output,
                    text=tokenizer.decode(output, skip_special_tokens=False),
                )
            )
    rotary = getattr(model.get_decoder(), "rotary_emb", None)
    effective_rope = dict(
        rope_type=getattr(rotary, "rope_type", None), attention_scaling=float(getattr(rotary, "attention_scaling", 1.0))
    )
    engine.save_tensor(root / "head.pt", model.get_output_embeddings().weight.detach().cpu())
    for i, row in enumerate(rows):
        path = root / f"{row['id']:04d}.pt"
        if not path.exists():
            value = hf_models.hidden(model, [row], pad_id=study.evaluation.stop_ids[0]).cpu()
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
        mode_probes=mode_probes,
        effective_rope=effective_rope,
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
    value = dict(
        student=study.models[study.student].identity,
        teacher=study.models[arm.teacher].identity,
        bank=teacher_inputs(study, arm.teacher)["bank"],
        training=asdict(study.training),
        arm=asdict(arm),
    )
    if study.continuation is not None:
        value["fixed_pool_batches"] = study.continuation.get("pool_batches", 20)
        origin = study.continuation.get("resume_from", {}).get(arm.name)
        if origin:
            saved = read(Path(origin) / "resume/complete.json")
            value["resume_origin"] = dict(
                step=saved["step"], weight_identity=saved["weight_identity"], signature=saved["signature"]
            )
    return value


def validation_teacher(study):
    from transformers import AutoConfig

    def layout(name):
        config = AutoConfig.from_pretrained(study.models[name].directory, local_files_only=True)
        return {key: getattr(config, key, None) for key in models.PARAMETER_LAYOUT_FIELDS}

    student = layout(study.student)
    requested = (study.continuation or {}).get("validation_teacher")
    if requested:
        if requested not in study.models or not models.same_parameter_layout(student, layout(requested)):
            raise ValueError("Numerical reference must share the student parameter layout")
        return requested
    matched = [a.teacher for a in study.arms if models.same_parameter_layout(student, layout(a.teacher))]
    return next(
        (name for name in matched if study.models[name].method.startswith("rl")),
        matched[0] if matched else study.student,
    )


def reference_arm(study, teacher):
    if teacher != validation_teacher(study):
        raise ValueError("Unused teacher is not the declared numerical reference")
    count = (study.continuation or {}).get("pool_batches", 20)
    return Arm("numerical-reference", teacher, updates=count * study.training.reuse, evaluate_at=())


def resume_signature(study, arm, execution):
    value = scientific_inputs(study, arm)
    value.pop("arm")
    value.pop("resume_origin", None)
    value["execution"] = asdict(execution)
    return value


def validate_checkpoint(path, *, verify_hash=False):
    """Check readout identity and presence; full content hashing is optional for offline audits."""
    path = Path(path)
    saved = read(path / "complete.json")
    if saved["fingerprint"] != fingerprint(saved["inputs"]):
        raise ValueError("Checkpoint input receipt changed")
    transient = saved["inputs"]["step"] != saved["inputs"]["arm"]["updates"]
    if (
        saved.get("transient") != transient
        or saved.get("dtype") != "float32"
        or saved.get("optimizer_saved") is not False
    ):
        raise ValueError("Checkpoint retention or precision contract changed")
    pruning = bool(saved.get("prune_started_at") or saved.get("weights_pruned"))
    if pruning and not transient:
        raise ValueError("Final checkpoint cannot be pruned")
    files = saved["files"]
    if not files or len({item["name"] for item in files}) != len(files):
        raise ValueError("Empty or duplicate checkpoint manifest")
    content = fingerprint(dict(weights=files, config=file_hash(path / "config.json")))
    if content != saved["weight_identity"]:
        raise ValueError("Checkpoint configuration or manifest changed")
    if verify_hash and pruning:
        raise ValueError("Pruned checkpoint cannot supply a new evaluation")
    for item in files:
        name = item["name"]
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("Unsafe checkpoint filename")
        source = path / name
        if not source.exists():
            if pruning:
                continue
            raise ValueError(f"Checkpoint weight missing: {source}")
        if source.stat().st_size != item["bytes"]:
            raise ValueError(f"Checkpoint size changed: {source}")
        if verify_hash and file_hash(source) != item["sha256"]:
            raise ValueError(f"Checkpoint content changed: {source}")
    return saved


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
    from examples.mirroropd.research.studies.fixed_transfer.validation import profile_directory

    near = validation_teacher(study)
    validated = read(profile_directory(study, near) / "complete.json")
    execution = Execution(**validated["execution"])
    target_config = read(study.models[arm.teacher].directory / "config.json")
    reference_config = read(study.models[near].directory / "config.json")
    extra_head = (
        max(
            0,
            (
                target_config["vocab_size"] * target_config["hidden_size"]
                - reference_config["vocab_size"] * reference_config["hidden_size"]
            )
            * 4,
        )
        / 2**30
        + 0.25
    )
    if validated.get("teacher_head_reserve_gib", 0) + 1e-8 < extra_head:
        raise ValueError("Numerical profile does not reserve enough memory for this teacher head")

    inputs = dict(
        scientific_inputs(study, arm), execution=asdict(execution), numerical_profile=validated["fingerprint"]
    )
    if complete(directory, inputs):
        return
    origin = (study.continuation or {}).get("resume_from", {}).get(arm.name)
    if (directory / "training.json").exists():
        raise RuntimeError("Partial scientific run exists; preserve it and use a distinct attempt")
    rows = required_rows(study, arm)
    if study.continuation is None:
        assert arm.updates // study.training.reuse <= len({r["group"] for r in rows if r["split"] == "train"})
    random.seed(study.training.seed)
    torch.manual_seed(study.training.seed)
    head, states = teacher_state(study, arm.teacher, rows)
    restored = read(Path(origin) / "resume/complete.json") if origin else None
    if restored:
        gate = read(Path(origin) / "resume/gpu-verified.json")
        if (
            (gate.get("signature"), gate.get("weight_identity"), gate.get("step"))
            != (restored["signature"], restored["weight_identity"], restored["step"])
            or not gate.get("exact_loaded_values")
            or not gate.get("rng_exact")
        ):
            raise ValueError("Resume checkpoint requires a matching GPU acceptance receipt")
        prior_profile = read(Path(origin) / "protocol.json")["inputs"]["numerical_profile"]
        if prior_profile != validated["fingerprint"]:
            raise ValueError("Resume requires the same verified numerical profile")
        if restored["signature"] != fingerprint(resume_signature(study, arm, execution)):
            raise ValueError("Resume science or execution signature changed")
        validate_checkpoint(Path(restored["weights"]), verify_hash=True)
    model = hf_models.load(study.models[study.student], trainable=True, path=restored["weights"] if restored else None)
    hf_models.training_mode(model, execution)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=study.training.learning_rate,
        betas=study.training.betas,
        eps=study.training.epsilon,
        weight_decay=study.training.weight_decay,
        foreach=False,
    )
    start_step = 0
    history = []
    if origin:
        start_step = training_state.restore(
            Path(origin) / "resume",
            model,
            optimizer,
            signature=resume_signature(study, arm, execution),
            weight_identity=restored["weight_identity"],
        )
        if start_step >= arm.updates:
            raise ValueError("Resume target must exceed prior completed stage")
        history = read(Path(origin) / "training.json")
        if len(history) != start_step or history[-1]["step"] != start_step:
            raise ValueError("Resume history does not match optimizer step")
        directory.mkdir(parents=True, exist_ok=True)
        for source in Path(origin).glob("fixed-*.json"):
            shutil.copy2(source, directory / source.name)
    write(
        directory / "protocol.json",
        dict(
            inputs=inputs,
            execution=asdict(execution),
            parameter_dtype="float32",
            optimizer_dtype="float32",
            rollout_source="fixed S0",
            save_policy="final/readout weights; rolling Adam state"
            if (study.continuation or {}).get("save_resume")
            else "final/readout weights; no Adam",
        ),
    )
    pad = study.evaluation.stop_ids[0]
    if not origin:
        write(directory / "fixed-000.json", engine.score(model, head, states, rows, 0, study.training, execution, pad))
    qcache = None
    score_steps = study.training.score_steps(arm.updates, study.risk_readouts(arm))
    stage_started = time.time()
    started = stage_started - (history[-1]["elapsed"] if history else 0)
    for step in range(start_step + 1, arm.updates + 1):
        group = study.training_group(step)
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
        if step in arm.evaluate_at or step == arm.updates:
            assert {int(s["step"]) for s in optimizer.state.values()} == {step}
            weight_identity = export_weights(study, arm, step, model, inputs)
            if step == arm.updates and (study.continuation or {}).get("save_resume"):
                training_state.save(
                    directory / "resume",
                    model,
                    optimizer,
                    step=step,
                    signature=resume_signature(study, arm, execution),
                    weights=checkpoint_directory(study, arm, step),
                    weight_identity=weight_identity,
                )
    receipt(
        directory,
        inputs,
        updates=arm.updates,
        added_updates=arm.updates - start_step,
        stage_elapsed=time.time() - stage_started,
        elapsed=time.time() - started,
    )
    if origin and (study.continuation or {}).get("save_resume"):
        training_state.retire(Path(origin) / "resume", replacement=directory / "resume")
    del model, optimizer, head, states
    gc.collect()


def evaluate_endpoint(study, name, step):
    from examples.mirroropd.research.studies.fixed_transfer.evaluation import evaluate, evaluation_inputs

    arm = arm_by_name(study, name)
    path = checkpoint_directory(study, arm, step)
    saved = read(path / "complete.json")
    expected = evaluation_inputs(study, study.student, saved["weight_identity"])
    evaluation_directory = study.cache_directory / "evaluations" / fingerprint(expected)
    if not complete(evaluation_directory, expected):
        if saved.get("weights_pruned"):
            raise RuntimeError("This transient readout was pruned after evaluation; regenerate it for new evaluation")
        validate_checkpoint(path, verify_hash=True)
    evaluate(study, study.student, weights=path, weight_identity=saved["weight_identity"], label=f"{name}-{step:03d}")


def endpoint_jobs(study, arm=None):
    """Build independent readout jobs without materializing teacher caches or numerical jobs."""
    result = []
    selected = [arm] if arm is not None else study.arms
    for arm in selected:
        inputs = scientific_inputs(study, arm)
        for step in arm.evaluate_at:
            result.append(
                dict(
                    name=f"eval-{arm.name}-{step:03d}",
                    args=["endpoint", "--arm", arm.name, "--step", str(step)],
                    dependencies=[],
                    priority=50,
                    ready_file=str(checkpoint_directory(study, arm, step) / "complete.json"),
                    ready_inputs=dict(inputs, step=step),
                    producer="train-" + arm.name,
                    completion_weight_from_ready=True,
                    marker=str(study.directory / "evaluations" / f"{arm.name}-{step:03d}/complete.json"),
                    retries=2,
                )
            )
    return result


def training_jobs(study, existing):
    from examples.mirroropd.research.studies.fixed_transfer.validation import (
        copy_inputs,
        profile_directory,
        profile_inputs,
    )

    result = []
    if not study.arms:
        return result
    banks = [f"bank-{i}" for i in range(4)]
    near = validation_teacher(study)
    for name in sorted({a.teacher for a in study.arms} | {near}):
        cache_directory = teacher_directory(study, name)
        result.append(
            dict(
                name="cache-" + name,
                args=["cache", "--model", name],
                dependencies=banks,
                marker=str(cache_directory / "complete.json"),
                fingerprint=fingerprint(teacher_inputs(study, name)),
                required_files=[
                    str(cache_directory / f"{r['id']:04d}.pt")
                    for r in required_rows(
                        study, next((a for a in study.arms if a.teacher == name), None) or reference_arm(study, name)
                    )
                ],
                retries=1,
            )
        )
    near = validation_teacher(study)
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
                ]
                + (
                    ["baseline-" + study.student, "baseline-" + arm.teacher]
                    if (study.continuation or {}).get("accuracy_gate", True)
                    else []
                ),
                marker=str(arm_directory(study, arm) / "complete.json"),
                expected_inputs=scientific_inputs(study, arm),
                retries=0,
            )
        )
        result.extend(endpoint_jobs(study, arm))
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
