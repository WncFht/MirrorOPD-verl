# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Validated continuation/new-teacher arms, final weights only, no optimizer exports."""

import argparse
import gc
import importlib.metadata
import math
import os
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from examples.mirroropd.research.backends import hf_rkl
from examples.mirroropd.research.core import training_state
from examples.mirroropd.research.core.artifacts import file_hash, fingerprint
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.boundary import fast_engine as engine
from examples.mirroropd.research.studies.boundary.native import bank, load, setup
from examples.mirroropd.research.studies.boundary.round2_common import (
    ARMS,
    GROUPS,
    MODELS,
    OLD_OUT,
    OUT,
    final_path,
    initial_path,
    model_path,
)


def restore(optimizer, arm, rows):
    config = ARMS[arm]
    state = torch.load(initial_path(arm) / "optimizer.pt", map_location="cpu", weights_only=False)
    assert state["step"] == 64 and state["arm"] == config["old_arm"]
    assert tuple(state["models"]) == ("r1", config["teacher"])
    assert all(tuple(state["model_revisions"][n]) == MODELS[n] for n in ["r1", config["teacher"]])
    historical = [r for r in rows if r["group"] in GROUPS]
    assert state["trajectories"] == [
        {k: row[k] for k in ["id", "prompt_ids", "response_ids", "source", "seed"]} for row in historical
    ]
    optimizer.load_state_dict(state["optimizer"])
    assert {int(s["step"]) for s in optimizer.state.values()} == {64}
    for group in optimizer.param_groups:
        assert group["lr"] == 1e-6 and group["betas"] == (0.9, 0.999)
        assert group["eps"] == 1e-8 and group["weight_decay"] == 0.01
    torch.set_rng_state(state["torch_rng"])
    assert len(state["cuda_rng"]) == 1
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    return dict(
        source=str(initial_path(arm)),
        optimizer_steps=[64],
        trajectories_checked=len(historical),
        rng_restored=True,
        optimizer_reset=False,
    )


def extension_plan(path, arm):
    value = read(path)
    if set(value["arms"]) != set(ARMS) or set(value["models"]) != {"r1", *(a["teacher"] for a in ARMS.values())}:
        raise ValueError("Extension must pin the six authorized teachers and original student")
    if value["models"] != {k: list(MODELS[k]) for k in value["models"]}:
        raise ValueError("Extension model revisions changed")
    expected = dict(
        batch_size=16,
        reuse=16,
        horizon=1024,
        train_groups=list("ABCDEFGHIJKLMNOPQRST"),
        learning_rate=1e-6,
        betas=[0.9, 0.999],
        eps=1e-8,
        weight_decay=0.01,
        clip=1,
        seed=42,
        engine="validated",
    )
    if value["training"] != expected or file_hash(value["source_protocol"]) != value["source_protocol_sha256"]:
        raise ValueError("Extension training settings or historical protocol changed")
    if value["scoring"] != dict(
        groups=list("ABCDEFGHIJKLMNOPQRST") + ["M01", "M02", "M03", "M04", "Code01", "Code02", "IF01", "IF02"],
        interval=16,
    ):
        raise ValueError("Extension must retain the fixed 448-prompt scoring panel")
    config = value["arms"][arm]
    old = ARMS[arm]
    replica = value.get("execution_condition") == "deterministic_from_initial"
    if replica:
        proof = read(value["repeatability_probe"])
        if file_hash(value["repeatability_probe"]) != value["repeatability_sha256"]:
            raise ValueError("Repeatability evidence changed")
        if proof["engine"] != file_hash(engine.__file__) or proof["backend"] != file_hash(hf_rkl.__file__):
            raise ValueError("Repeatability execution code changed")
        if proof["versions"] != {n: importlib.metadata.version(n) for n in proof["versions"]}:
            raise ValueError("Repeatability package versions changed")
        controls = {p["deterministic"]: p for p in proof["results"]}
        if (
            proof["student"] != list(MODELS["r1"])
            or proof["scientific_updates"] != 0
            or controls[False]["unequal_gradient_tensors"] == 0
            or controls[True]["unequal_gradient_tensors"] != 0
            or controls[True]["relative_l2"] != 0
        ):
            raise ValueError("Deterministic replica requires measured repeatability evidence")
        if config["start"] != 0 or config["old_arm"] is not None or config["teacher"] != old["teacher"]:
            raise ValueError("All deterministic replicas must begin at the original student")
    elif any(config[k] != old[k] for k in ["teacher", "old_arm", "start"]):
        raise ValueError("Extension must preserve the existing initialization")
    if config["stop"] != 320 or config["replay_through"] != old["stop"]:
        raise ValueError("This extension requires replay acceptance followed by 320 updates")
    if config["new_updates"] != (320 if replica else 320 - old["stop"]):
        raise ValueError("Replay work cannot be counted as new scientific updates")
    expected_source = model_path("r1") if replica else initial_path(arm)
    if Path(config["source_weights"]).resolve() != expected_source.resolve():
        raise ValueError("Extension source weights changed")
    if Path(value["output"]).resolve() in {OUT.resolve(), OLD_OUT.resolve()}:
        raise ValueError("Extension cannot overwrite historical science")
    return value


def deterministic_gate(model, head, states, rows, chunk, destination):
    from types import SimpleNamespace

    from examples.mirroropd.research.studies.boundary.small_to_large import grad_compare

    assert os.environ.get("FLASH_ATTENTION_DETERMINISTIC") == "1"
    sample = [next(r for r in rows if r["group"] == "A")]
    q = hf_rkl.log_probabilities(head, states, sample, chunk)
    model.zero_grad(set_to_none=True)
    reference = hf_rkl.reference_backward(model, head, states, sample, SimpleNamespace(denominator=16 * 1024), 151643)
    saved = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    first = engine.backward(model, q, sample, chunk)
    comparison = grad_compare(model, saved)
    del saved
    assert abs(first - reference) < 1e-5 and comparison["relative_l2"] < 0.01 and comparison["cosine"] > 0.9999
    saved = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    second = engine.backward(model, q, sample, chunk)
    unequal = sum(not torch.equal(saved[n], p.grad.detach().cpu()) for n, p in model.named_parameters())
    assert first == second and unequal == 0
    dump(
        destination,
        dict(
            reference_loss=reference,
            candidate_loss=first,
            repeat_loss=second,
            unequal_gradient_tensors=unequal,
            scientific_updates=0,
            prompts=1,
            **comparison,
        ),
    )
    model.zero_grad(set_to_none=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("--engine", choices=["validated", "batched"], default="validated")
    parser.add_argument("--plan", type=Path, help="Isolated 320-update extension with audited replay")
    args = parser.parse_args()
    setup()
    random.seed(42)
    torch.manual_seed(42)
    extension = extension_plan(args.plan, args.arm) if args.plan else None
    replica = bool(extension and extension.get("execution_condition") == "deterministic_from_initial")
    if extension:
        os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1" if replica else "0"
    if extension and args.engine != extension["training"]["engine"]:
        raise ValueError("Replay must retain the original validated execution mode")
    config = extension["arms"][args.arm] if extension else ARMS[args.arm]
    dest = (Path(extension["output"]) if extension else OUT) / "arms" / args.arm
    saved = Path(extension["cache"]) / "models" / args.arm / "step320" if extension else final_path(args.arm)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "complete.json").exists():
        if extension and read(dest / "complete.json").get("plan_identity") != fingerprint(extension):
            raise ValueError("Completed extension belongs to another plan")
        return
    if (dest / "training.json").exists():
        raise RuntimeError("Partial scientific run exists; preserve before an explicit rerun")
    optimization = read(
        OUT / ("optimization-batched/complete.json" if args.engine == "batched" else "optimization/complete.json")
    )
    chunk = optimization["chunk"]
    rows = [r for r in bank("r1") if r["group"] in extension["scoring"]["groups"]] if extension else engine.panel()
    head, states = engine.teacher_cache(
        config["teacher"],
        rows,
        cache_root=extension.get("teacher_cache_root", extension["cache"]) if extension else None,
    )
    if extension and config["start"] > 0:
        original = load("r1", student=True)
        dump(dest / "fixed-000.json", engine.score(original, head, states, rows, 0, chunk))
        del original
        gc.collect()
        torch.cuda.empty_cache()
    model = load("r1", student=True, path=Path(config["source_weights"]) if extension else initial_path(args.arm))
    engine.training_mode(model)
    if args.engine == "batched":
        model.gradient_checkpointing_disable()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False
    )
    resume = restore(optimizer, args.arm, rows) if config["old_arm"] else dict(optimizer_reset=False, fresh=True)
    protocol = read(OUT / "protocol.json")
    if extension:
        protocol.update(
            arms=extension["arms"],
            scope=extension.get("interpretation", "R1 replay plus extension to 320"),
            checkpoint_policy="final FP32 weights and Adam/RNG; no intermediate model exports",
        )
    dump(
        dest / "protocol.json",
        protocol
        | dict(
            arm=args.arm,
            **config,
            resume=resume,
            optimization=optimization,
            engine_mode=args.engine,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            final_weights=str(saved),
            weights_dtype="FP32",
            optimizer_exports=bool(extension),
            extension_plan=extension,
            flash_attention_deterministic=replica,
        ),
    )
    initial = engine.score(model, head, states, rows, config["start"], chunk)
    dump(dest / f"fixed-{config['start']:03d}.json", initial)
    if replica:
        deterministic_gate(model, head, states, rows, chunk, dest / "deterministic-gate.json")
    if config["old_arm"]:
        old = read(OLD_OUT / "arms" / config["old_arm"] / "fixed-064.json")
        initial_by_id = {r["id"]: r for r in initial}
        error = max(abs(a["stats"]["all"]["rkl"] - initial_by_id[a["id"]]["stats"]["all"]["rkl"]) / 1024 for a in old)
        dump(dest / "resume-score-check.json", dict(max_per_question_rkl_difference=error, tolerance=1e-4))
        assert error < 1e-4, error
    history = []
    started = time.time()
    qcache = None
    for step in range(config["start"] + 1, config["stop"] + 1):
        group = chr(65 + (step - 1) // 16)
        train_rows = [r for r in rows if r["group"] == group]
        assert len(train_rows) == 16 and all(r["split"] == "train" for r in train_rows)
        cache_seconds = 0.0
        if (step - 1) % 16 == 0:
            del qcache
            cache_start = time.perf_counter()
            qcache = hf_rkl.log_probabilities(head, states, train_rows, chunk)
            torch.cuda.synchronize()
            cache_seconds = time.perf_counter() - cache_start
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = (
            engine.backward(model, qcache, train_rows, chunk, optimization["microbatch"])
            if args.engine == "batched"
            else engine.backward(model, qcache, train_rows, chunk)
        )
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
        assert math.isfinite(loss) and bool(torch.isfinite(norm))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        record = dict(
            step=step,
            group=group,
            loss=loss,
            grad_norm=float(norm),
            actual_tokens=sum(len(r["response_ids"]) for r in train_rows),
            cache_seconds=cache_seconds,
            backward_seconds=t1 - t0,
            optimizer_seconds=t2 - t1,
            update_seconds=t2 - t0,
            elapsed=time.time() - started,
            peak_gib=torch.cuda.max_memory_allocated() / 2**30,
        )
        history.append(record)
        dump(dest / "training.json", history)
        print("UPDATE", args.arm, record, flush=True)
        optimizer.zero_grad(set_to_none=True)
        if step % 16 == 0:
            del qcache
            qcache = None
            dump(dest / f"fixed-{step:03d}.json", engine.score(model, head, states, rows, step, chunk))
            if extension and step == config["replay_through"]:
                reference = read(OUT / "arms" / args.arm / f"fixed-{step:03d}.json")
                current = {r["id"]: r for r in read(dest / f"fixed-{step:03d}.json")}
                risk_error = max(
                    abs(r["stats"]["all"]["rkl"] - current[r["id"]]["stats"]["all"]["rkl"]) / 1024 for r in reference
                )
                old_history = read(OUT / "arms" / args.arm / "training.json")
                old_steps = {r["step"] for r in old_history}
                matched = [r for r in history if r["step"] in old_steps]
                assert [r["step"] for r in old_history] == [r["step"] for r in matched]
                loss_error = max(abs(a["loss"] - b["loss"]) for a, b in zip(matched, old_history, strict=True))
                gate = extension["replay_acceptance"]
                dump(
                    dest / "replay-acceptance.json",
                    dict(
                        step=step,
                        risk_max_error=risk_error,
                        loss_max_error=loss_error,
                        old_prompts=len(reference),
                        optimizer_reset=False,
                        reference_policy="diagnostic" if replica else "strict",
                        exact_old_trajectory_claim=not replica,
                    ),
                )
                if not replica and (
                    risk_error > gate["per_prompt_rkl_absolute_tolerance"]
                    or loss_error > gate["max_loss_absolute_tolerance"]
                ):
                    raise ValueError("Replay differs from historical results; no new-budget updates permitted")
    assert {int(s["step"]) for s in optimizer.state.values()} == {config["stop"]}
    saved.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        saved, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, max_shard_size="4GB"
    )
    AutoTokenizer.from_pretrained(model_path("r1"), local_files_only=True).save_pretrained(saved)
    files = [
        dict(name=f.name, bytes=f.stat().st_size, sha256=file_hash(f)) for f in sorted(saved.glob("*.safetensors"))
    ]
    weight_identity = fingerprint(dict(weights=files, config=file_hash(saved / "config.json")))
    dump(
        saved / "complete.json",
        dict(
            arm=args.arm,
            step=config["stop"],
            optimizer_saved=bool(extension),
            dtype="FP32",
            files=files,
            weight_identity=weight_identity,
        ),
    )
    if extension:
        training_state.save(
            dest / "resume",
            model,
            optimizer,
            step=config["stop"],
            signature=dict(
                models=extension["models"],
                training=extension["training"],
                teacher=config["teacher"],
                bank=fingerprint(rows),
                **(dict(execution_condition=extension["execution_condition"]) if replica else {}),
            ),
            weights=saved,
            weight_identity=weight_identity,
        )
    dump(
        dest / "complete.json",
        dict(
            arm=args.arm,
            start=config["start"],
            stop=config["stop"],
            updates=len(history),
            elapsed=time.time() - started,
            weights=str(saved),
            plan_identity=fingerprint(extension) if extension else None,
            inputs=dict(plan_identity=fingerprint(extension)) if extension else {},
            new_updates=config["new_updates"] if extension else len(history),
        ),
    )
    del optimizer, model
    gc.collect()


if __name__ == "__main__":
    main()
