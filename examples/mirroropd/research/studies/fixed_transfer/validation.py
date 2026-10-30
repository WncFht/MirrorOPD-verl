# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Numerical gates, worst-batch execution tuning and feasible teacher-copy references."""

import gc
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from examples.mirroropd.research.backends import hf_models
from examples.mirroropd.research.backends import hf_rkl as engine
from examples.mirroropd.research.core.artifacts import complete, file_hash, receipt
from examples.mirroropd.research.core.gradient_audit import compare, measure, snapshot
from examples.mirroropd.research.core.io import dump as write
from examples.mirroropd.research.core.io import read
from examples.mirroropd.research.studies.fixed_transfer import models
from examples.mirroropd.research.studies.fixed_transfer.data import required_rows
from examples.mirroropd.research.studies.fixed_transfer.spec import fingerprint
from examples.mirroropd.research.studies.fixed_transfer.training import reference_arm, teacher_inputs, teacher_state


def copy_parameters(student, teacher):
    sp = dict(student.named_parameters(remove_duplicate=False))
    tp = dict(teacher.named_parameters(remove_duplicate=False))
    assert sp.keys() == tp.keys()
    if student.get_input_embeddings().weight is student.get_output_embeddings().weight:
        assert torch.equal(teacher.get_input_embeddings().weight, teacher.get_output_embeddings().weight)
    sums = torch.zeros(3, device="cuda", dtype=torch.float64)
    with torch.no_grad():
        # Iterate unique student parameters so tied matrices are counted once.
        for name, p in student.named_parameters():
            t = tp[name].float()
            assert p.shape == t.shape
            sums += torch.stack(
                [
                    p.square().sum(dtype=torch.float64),
                    t.square().sum(dtype=torch.float64),
                    (p - t).square().sum(dtype=torch.float64),
                ]
            )
            p.copy_(t)
    a, b, delta = sums.tolist()
    return dict(
        relative_parameter_l2=(delta / a) ** 0.5,
        student_norm=a**0.5,
        teacher_norm=b**0.5,
        parameter_distance=delta**0.5,
        buffers_copied=False,
    )


def profile_inputs(study, teacher):
    return dict(
        student=study.models[study.student].identity,
        teacher=study.models[teacher].identity,
        bank=teacher_inputs(study, teacher)["bank"],
        training=asdict(study.training),
        execution=asdict(study.execution),
        engine_code=file_hash(engine.__file__),
        model_code=file_hash(hf_models.__file__),
        interface_code=file_hash(models.__file__),
        torch=torch.__version__,
    )


def profile_directory(study, teacher):
    return study.directory / "numerical" / fingerprint(profile_inputs(study, teacher))


def inherit_resume_profile(study, teacher):
    """Keep the validated execution used by saved Adam; preserve any discarded retuning evidence."""
    origins = (study.continuation or {}).get("resume_from", {})
    if not origins:
        return False
    from examples.mirroropd.research.studies.fixed_transfer.spec import Execution
    from examples.mirroropd.research.studies.fixed_transfer.training import resume_signature

    inputs = profile_inputs(study, teacher)
    source, selected = None, None
    for arm in study.arms:
        if arm.name not in origins:
            continue
        origin = Path(origins[arm.name])
        protocol = read(origin / "protocol.json")
        candidate = origin.parent.parent / "numerical" / protocol["inputs"]["numerical_profile"]
        value = read(candidate / "complete.json")
        if fingerprint(value["inputs"]) != fingerprint(inputs) or value["fingerprint"] != fingerprint(inputs):
            raise ValueError("Resume numerical inputs differ; cannot inherit the earlier profile")
        if value["execution"] != protocol["execution"]:
            raise ValueError("Source execution differs from its numerical evidence")
        saved = read(origin / "resume/complete.json")
        expected = resume_signature(study, arm, Execution(**value["execution"]))
        if saved["signature"] != fingerprint(expected):
            raise ValueError("Resume science changed even with the original execution")
        if selected is not None and value["execution"] != selected["execution"]:
            raise ValueError("Different resume execution profiles require separate studies")
        source, selected = candidate, value
    if source is None:
        return False
    reference = read(study.models[teacher].directory / "config.json")
    configs = [read(study.models[a.teacher].directory / "config.json") for a in study.arms]
    largest = max(c["vocab_size"] * c["hidden_size"] * 4 for c in configs)
    reserve = max(0, largest - reference["vocab_size"] * reference["hidden_size"] * 4) / 2**30 + 0.25
    if selected.get("teacher_head_reserve_gib", 0) + 1e-8 < reserve:
        raise ValueError("Inherited profile does not cover the requested teacher head")
    destination = profile_directory(study, teacher)
    if complete(destination, inputs) and read(destination / "complete.json")["execution"] == selected["execution"]:
        return True
    if any((study.directory / "arms" / arm.name / "training.json").exists() for arm in study.arms):
        raise ValueError("Cannot replace execution evidence after scientific updates")
    archived = None
    if destination.exists():
        archived = destination.with_name(destination.name + f"-retuning-{time.time_ns()}")
        destination.rename(archived)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    write(
        destination / "inheritance.json",
        dict(
            source=str(source),
            source_sha256=file_hash(source / "complete.json"),
            archived_retuning=str(archived) if archived else None,
            execution=selected["execution"],
            scientific_updates=0,
        ),
    )
    return True


def validate(study, teacher):
    if inherit_resume_profile(study, teacher):
        return
    inputs = profile_inputs(study, teacher)
    destination = profile_directory(study, teacher)
    destination.mkdir(parents=True, exist_ok=True)
    if complete(destination, inputs):
        return
    arm = next((a for a in study.arms if a.teacher == teacher), None)
    if arm is None:
        arm = reference_arm(study, teacher)
    rows = required_rows(study, arm)
    groups = {r["group"] for r in rows if r["split"] == "train"}
    batches = [[r for r in rows if r["group"] == g] for g in sorted(groups)]
    batch = max(batches, key=lambda rs: max(len(r["prompt_ids"]) + len(r["response_ids"]) for r in rs))
    assert len(batch) == study.training.batch_size
    head, states = teacher_state(study, teacher, rows)
    qcache = engine.log_probabilities(head, states, batch, 512)
    configs = [read(study.models[a.teacher].directory / "config.json") for a in study.arms]
    largest_head = max(c["vocab_size"] * c["hidden_size"] * 4 for c in configs)
    teacher_reserve_gib = max(0, largest_head - head.numel() * head.element_size()) / 2**30 + 0.25
    evidence = dict(
        thresholds=dict(loss_absolute=1e-5, gradient_relative_l2=0.01, cosine=0.9999, peak_with_adam_gib=74),
        batch_ids=[r["id"] for r in batch],
        probes=[],
        teacher=teacher,
        teacher_head_reserve_gib=teacher_reserve_gib,
    )
    write(destination / "thresholds.json", evidence)
    pad = study.evaluation.stop_ids[0]
    for point in ["S0", "teacher-copy"]:
        model = hf_models.load(study.models[study.student], trainable=True)
        if point == "teacher-copy":
            teacher_model = hf_models.load(study.models[teacher])
            evidence["copy_distance"] = copy_parameters(model, teacher_model)
            del teacher_model
            gc.collect()
            torch.cuda.empty_cache()
        hf_models.training_mode(model, study.execution)
        measure(model, lambda m=model: engine.reference_backward(m, head, states, batch[:1], study.training, pad))
        reference = measure(
            model, lambda m=model: engine.reference_backward(m, head, states, batch, study.training, pad)
        )
        gradients = snapshot(model)
        candidates = []
        for mb in [1, 2, 4, 8, 16]:
            if study.training.batch_size % mb:
                continue
            execution = replace(study.execution, microbatch=mb, activation_checkpointing=False)
            hf_models.training_mode(model, execution)
            try:
                value = measure(
                    model, lambda m=model, e=execution: engine.backward(m, qcache, batch, study.training, e, pad)
                )
                value.update(compare(model, gradients))
                value["loss_error"] = abs(value["loss"] - reference["loss"])
                value["estimated_peak_with_adam_gib"] = (
                    value["peak_gib"] + sum(p.numel() * 8 for p in model.parameters()) / 2**30 + teacher_reserve_gib
                )
                accepted = (
                    value["loss_error"] < 1e-5
                    and value["relative_l2"] < 0.01
                    and value["cosine"] > 0.9999
                    and value["estimated_peak_with_adam_gib"] < 74
                )
                candidates.append(dict(microbatch=mb, accepted=accepted, **value))
            except torch.OutOfMemoryError:
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                candidates.append(dict(microbatch=mb, accepted=False, reason="out_of_memory"))
            print("NUMERICAL", point, candidates[-1], flush=True)
        evidence["probes"].append(dict(point=point, reference=reference, candidates=candidates))
        write(destination / "evidence.json", evidence)
        del model, gradients
        gc.collect()
        torch.cuda.empty_cache()
    accepted = [
        mb
        for mb in [1, 2, 4, 8, 16]
        if all(any(c["microbatch"] == mb and c["accepted"] for c in p["candidates"]) for p in evidence["probes"])
    ]
    if not accepted:
        raise RuntimeError("No numerical and memory-safe candidate passed")
    best = min(
        accepted,
        key=lambda mb: sum(
            next(c["seconds"] for c in p["candidates"] if c["microbatch"] == mb) for p in evidence["probes"]
        ),
    )
    execution = replace(study.execution, microbatch=best, activation_checkpointing=False)
    model = hf_models.load(study.models[study.student], trainable=True)
    hf_models.training_mode(model, execution)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=study.training.learning_rate,
        betas=study.training.betas,
        eps=study.training.epsilon,
        weight_decay=study.training.weight_decay,
        foreach=False,
    )
    smoke = []
    for index in range(2):
        samples = [
            (p, max(1, p.numel() // 1024), p.detach().reshape(-1)[:: max(1, p.numel() // 1024)][:1024].clone())
            for p in model.parameters()
        ]
        value = measure(model, lambda: engine.backward(model, qcache, batch, study.training, execution, pad))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), study.training.clip, foreach=False)
        assert bool(torch.isfinite(norm))
        start = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        value.update(
            optimizer_seconds=time.perf_counter() - start,
            grad_norm=float(norm),
            actual_peak_with_adam_gib=torch.cuda.max_memory_allocated() / 2**30,
        )
        delta = torch.zeros((), device="cuda", dtype=torch.float64)
        for parameter, stride, before in samples:
            delta += (parameter.detach().reshape(-1)[::stride][:1024] - before).square().sum(dtype=torch.float64)
        value["sampled_parameter_change_l2"] = float(delta.sqrt())
        assert value["sampled_parameter_change_l2"] > 0
        del samples
        assert value["actual_peak_with_adam_gib"] + teacher_reserve_gib < 74
        value["optimizer_steps"] = sorted({int(s["step"]) for s in optimizer.state.values()})
        assert value["optimizer_steps"] == [index + 1]
        smoke.append(value)
    evidence.update(
        execution=asdict(execution),
        accepted_microbatches=accepted,
        disposable_adam_steps=smoke,
        scientific_updates=0,
        checkpoint_exports=0,
    )
    receipt(destination, inputs, **evidence)
    write(study.directory / "numerical/latest.json", dict(directory=str(destination), teacher=teacher))


def copy_inputs(study, teacher):
    return dict(
        student=study.models[study.student].identity,
        target=teacher_inputs(study, teacher),
        parameter_copy="strict; original student buffers and configuration",
        contract=1,
    )


@torch.no_grad()
def teacher_copy(study, teacher):
    destination = study.directory / "teacher-copy" / teacher
    inputs = copy_inputs(study, teacher)
    if complete(destination, inputs):
        return
    arm = next((a for a in study.arms if a.teacher == teacher), None)
    if arm is None:
        arm = reference_arm(study, teacher)
    rows = [r for r in required_rows(study, arm) if r["group"] in ["M01", "M02", "M03", "M04"]]
    head, states = teacher_state(study, teacher, rows)
    model = hf_models.load(study.models[study.student], trainable=True)
    target = hf_models.load(study.models[teacher])
    distance = copy_parameters(model, target)
    del target
    gc.collect()
    torch.cuda.empty_cache()
    execution = replace(study.execution, microbatch=1)
    scored = engine.score(model, head, states, rows, 0, study.training, execution, study.evaluation.stop_ids[0])
    write(destination / "scores.json", scored)
    receipt(
        destination,
        inputs,
        **dict(
            teacher=teacher,
            **distance,
            training_updates=0,
            rkl=sum(r["stats"]["all"]["rkl"] / study.training.horizon for r in scored) / len(scored),
            js=sum(r["stats"]["all"]["js"] / study.training.horizon for r in scored) / len(scored),
            interpretation="Feasible parameter point in original student configuration; not a lower bound",
        ),
    )
