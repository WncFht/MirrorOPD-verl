# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Validated continuation/new-teacher arms, final weights only, no optimizer exports."""

import argparse
import gc
import math
import os
import random
import time

import torch
from transformers import AutoTokenizer

from examples.mirroropd.research.backends import hf_rkl
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.boundary import fast_engine as engine
from examples.mirroropd.research.studies.boundary.native import load, setup
from examples.mirroropd.research.studies.boundary.round2_common import (
    ARMS,
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
    assert state["trajectories"] == [
        {k: row[k] for k in ["id", "prompt_ids", "response_ids", "source", "seed"]} for row in rows
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
        trajectories_checked=len(rows),
        rng_restored=True,
        optimizer_reset=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("--engine", choices=["validated", "batched"], default="validated")
    args = parser.parse_args()
    setup()
    random.seed(42)
    torch.manual_seed(42)
    config = ARMS[args.arm]
    dest = OUT / "arms" / args.arm
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "complete.json").exists():
        return
    if (dest / "training.json").exists():
        raise RuntimeError("Partial scientific run exists; preserve before an explicit rerun")
    optimization = read(
        OUT / ("optimization-batched/complete.json" if args.engine == "batched" else "optimization/complete.json")
    )
    chunk = optimization["chunk"]
    rows = engine.panel()
    head, states = engine.teacher_cache(config["teacher"], rows)
    model = load("r1", student=True, path=initial_path(args.arm))
    engine.training_mode(model)
    if args.engine == "batched":
        model.gradient_checkpointing_disable()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False
    )
    resume = restore(optimizer, args.arm, rows) if config["old_arm"] else dict(optimizer_reset=False, fresh=True)
    dump(
        dest / "protocol.json",
        read(OUT / "protocol.json")
        | dict(
            arm=args.arm,
            **config,
            resume=resume,
            optimization=optimization,
            engine_mode=args.engine,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            final_weights=str(final_path(args.arm)),
            weights_dtype="FP32",
            optimizer_exports=False,
        ),
    )
    initial = engine.score(model, head, states, rows, config["start"], chunk)
    dump(dest / f"fixed-{config['start']:03d}.json", initial)
    if config["old_arm"]:
        old = read(OLD_OUT / "arms" / config["old_arm"] / "fixed-064.json")
        assert [r["id"] for r in old] == [r["id"] for r in initial]
        error = max(
            abs(a["stats"]["all"]["rkl"] - b["stats"]["all"]["rkl"]) / 1024 for a, b in zip(old, initial, strict=True)
        )
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
    assert {int(s["step"]) for s in optimizer.state.values()} == {config["stop"]}
    saved = final_path(args.arm)
    saved.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        saved, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, max_shard_size="4GB"
    )
    AutoTokenizer.from_pretrained(model_path("r1"), local_files_only=True).save_pretrained(saved)
    dump(saved / "complete.json", dict(arm=args.arm, step=config["stop"], optimizer_saved=False, dtype="FP32"))
    dump(
        dest / "complete.json",
        dict(
            arm=args.arm,
            start=config["start"],
            stop=config["stop"],
            updates=len(history),
            elapsed=time.time() - started,
            weights=str(saved),
        ),
    )
    del optimizer, model
    gc.collect()


if __name__ == "__main__":
    main()
