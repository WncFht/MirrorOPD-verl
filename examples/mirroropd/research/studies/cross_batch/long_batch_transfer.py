#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Two-GPU, single-pass A..T training with a complete 40-batch transfer matrix.

Each rank accumulates half of the global batch divided by GLOBAL B*H; SUM
all-reduce produces the global mean gradient. Held-out gradients are diagnostic
only, cleared before the training gradient, and never passed to AdamW.
"""

import argparse
import datetime
import gc
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.mirroropd.assets.paths import checkpoint_path, registered_model
from examples.mirroropd.research.backends import hf_native as core
from examples.mirroropd.research.core.cache import trajectory_identity
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.core.metrics import fixed_metrics as metrics
from examples.mirroropd.research.studies.cross_batch.protocol import LONG_PASS, validate_long_pass, validate_resume


def all_sum(value):
    tensor = torch.tensor(value, dtype=torch.float64, device="cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor)


def reduce_gradients(model):
    for param in model.parameters():
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)


def backward(model, head, hidden, rows, global_count):
    value = 0.0
    for row in rows:
        hp = core.hidden(model, row["prompt_ids"] + row["response_ids"][:-1])[len(row["prompt_ids"]) - 1 :]
        hq = hidden[row["id"]]
        terms = []
        for start in range(0, len(hp), 128):
            lp = torch.nn.functional.linear(hp[start : start + 128].float(), model.lm_head.weight).log_softmax(-1)
            with torch.no_grad():
                lq = torch.nn.functional.linear(hq[start : start + 128].float(), head).log_softmax(-1)[
                    :, : lp.shape[-1]
                ]
            terms.append((lp.exp() * (lp - lq)).sum() / (global_count * LONG_PASS.horizon))
        loss = sum(terms)
        loss.backward()
        value += float(loss.detach())
        del hp, hq, lp, lq, loss, terms
    reduce_gradients(model)
    return all_sum(value)


@torch.no_grad()
def score_local(model, head, hidden, rows, batch, rkl_only=False):
    result = []
    for row in rows:
        hp = core.hidden(model, row["prompt_ids"] + row["response_ids"][:-1])[len(row["prompt_ids"]) - 1 :]
        hq = hidden[row["id"]]
        bands = {name: dict(tokens=max(0, min(len(hp), hi) - lo)) for name, lo, hi in LONG_PASS.bands()}
        for start in range(0, len(hp), 128):
            sp = torch.nn.functional.linear(hp[start : start + 128].float(), model.lm_head.weight)
            tq = torch.nn.functional.linear(hq[start : start + 128].float(), head)
            if rkl_only:
                lp = sp.log_softmax(-1)
                lq = tq.log_softmax(-1)[:, : sp.shape[-1]]
                vals = dict(rkl=(lp.exp() * (lp - lq)).sum(-1))
            else:
                vals = metrics(sp, tq)
            for name, lo, hi in LONG_PASS.bands():
                left, right = max(lo - start, 0), min(hi - start, len(sp))
                if right <= left:
                    continue
                for key, value in vals.items():
                    bands[name][key] = bands[name].get(key, 0.0) + float(value[left:right].double().sum())
            del sp, tq, vals
        result.append(
            dict(
                id=row["id"],
                group=row["group"],
                batch_index=row["batch_index"],
                split=row["split"],
                domain=row["domain"],
                completed_batches=batch,
                step=batch * LONG_PASS.reuse,
                length=len(hp),
                stats=bands,
            )
        )
        del hp, hq
    return result


def collect(a, local, path=None):
    gathered = [None] * a.world if a.rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    result = None
    if a.rank == 0:
        result = sorted(sum(gathered, []), key=lambda r: (r["id"], r.get("seed", -1)))
        if path is not None:
            dump(path, result)
    dist.barrier()
    return result


@torch.no_grad()
def replicas_match(model):
    maximum = torch.zeros((), device="cuda")
    for param in model.parameters():
        copy = param.detach().clone()
        dist.broadcast(copy, src=0)
        maximum = torch.maximum(maximum, (copy - param).abs().max())
        del copy
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    assert float(maximum) == 0.0, "Student replicas diverged."
    return float(maximum)


def fresh(a, model, teacher, rows, batch, dest):
    teacher.cuda()
    torch.cuda.empty_cache()
    output = []
    prompts = [r for r in rows if r["group"] in ["M01", "M02"]]
    assert len(prompts) == 16
    for base_seed in [77101, 77102]:
        seed = base_seed + 1000 * a.rank
        for r in core.decode(a, model, teacher, prompts, "student", seed):
            output.append(
                dict(
                    r,
                    source=f"fresh-{a.teacher}-batch{batch:02d}-seed{seed}",
                    reference_source=r["source"],
                    completed_batches=batch,
                    step=batch * LONG_PASS.reuse,
                )
            )
    collect(a, output, dest / f"fresh-{batch:02d}.json")
    teacher.cpu()
    gc.collect()
    torch.cuda.empty_cache()


def checkpoint(a, model, optimizer, batch, dest):
    error = replicas_match(model)
    rng = dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), python_rng=random.getstate())
    rng_states = [None] * a.world if a.rank == 0 else None
    dist.gather_object(rng, rng_states, dst=0)
    if a.rank == 0:
        target = checkpoint_path(a.cache, "checkpoints") / a.teacher / f"batch-{batch:02d}"
        target.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(target, max_shard_size="4GB")
        AutoTokenizer.from_pretrained(
            registered_model(core.NAMES["student"], root=a.models), local_files_only=True
        ).save_pretrained(target)
        state = optimizer.state_dict()
        cpu_state = dict(
            param_groups=state["param_groups"],
            state={
                i: {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in item.items()}
                for i, item in state["state"].items()
            },
        )
        payload = dict(
            experiment=a.experiment,
            optimizer=cpu_state,
            completed_batches=batch,
            step=batch * LONG_PASS.reuse,
            rng_states=rng_states,
        )
        temporary = target / "optimizer.pt.tmp"
        torch.save(payload, temporary)
        temporary.replace(target / "optimizer.pt")
        dump(
            target / "complete.json",
            dict(completed_batches=batch, step=batch * LONG_PASS.reuse, replica_max_difference=error),
        )
        dump(
            dest / "latest-checkpoint.json",
            dict(path=str(target), completed_batches=batch, step=batch * LONG_PASS.reuse),
        )
        del state, cpu_state, payload
        print("CHECKPOINT", a.teacher, batch, str(target), flush=True)
    dist.barrier()


def run(a):
    dest = a.out / a.teacher
    dest.mkdir(exist_ok=True)
    assert not (dest / "complete.json").exists(), "Completed study exists."
    validate_long_pass(read(a.out / "protocol.json"))
    bank = read(a.out / "bank.json")
    a.experiment = dict(
        teacher=a.teacher,
        world_size=a.world,
        models=str(a.models.resolve()),
        settings=LONG_PASS.as_record(),
        bank=[trajectory_identity(row) for row in bank],
    )
    assert len(bank) == LONG_PASS.examples
    rows = [r for r in bank if r["id"] % a.world == a.rank]
    diagnostic = [r for r in rows if r["group"] in ["M01", "M02", "M03", "M04"]]
    assert len(rows) == 320 and len(diagnostic) == 32
    path = a.resume if a.resume else registered_model(core.NAMES["student"], root=a.models)
    model = (
        AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float32, attn_implementation="flash_attention_2", local_files_only=True
        )
        .cuda()
        .eval()
    )
    teacher = core.load(a, a.teacher)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LONG_PASS.lr, betas=LONG_PASS.betas, weight_decay=LONG_PASS.weight_decay, foreach=True
    )
    start_batch = 0
    if a.resume:
        payload = torch.load(a.resume / "optimizer.pt", map_location="cpu", weights_only=True)
        validate_resume(payload, a.experiment, a.stop_after_batches)
        optimizer.load_state_dict(payload["optimizer"])
        start_batch = payload["completed_batches"]
        rng = payload["rng_states"][a.rank]
        torch.set_rng_state(rng["torch_rng"])
        torch.cuda.set_rng_state(rng["cuda_rng"])
        random.setstate(rng["python_rng"])
        del payload
    elif (dest / "training.json").exists():
        raise RuntimeError("Existing partial run requires an explicit --resume checkpoint.")
    head = teacher.lm_head.weight.float()
    hidden = {}
    with torch.no_grad():
        for i, r in enumerate(rows, 1):
            hidden[r["id"]] = core.teacher_hidden(a, teacher, r)
            if i % 64 == 0:
                print("CACHE", a.teacher, a.rank, i, "/", len(rows), flush=True)
    teacher.cpu()
    torch.cuda.empty_cache()
    replicas_match(model)
    history = []
    probes = []
    if a.rank == 0:
        if start_batch:
            history = [r for r in read(dest / "training.json") if r["step"] <= LONG_PASS.reuse * start_batch]
            probes = [r for r in read(dest / "update-probes.json") if r["training_batch"] <= start_batch]
        dump(
            dest / "protocol.json",
            read(a.out / "protocol.json")
            | dict(
                teacher=a.teacher,
                world_size=a.world,
                parameters=sum(p.numel() for p in model.parameters()),
                optimizer_state_preserved_between_batches=True,
            ),
        )
    if start_batch == 0:
        previous = collect(a, score_local(model, head, hidden, rows, 0), dest / "fixed-00.json")
        fresh(a, model, teacher, rows, 0, dest)
    else:
        previous = read(dest / f"fixed-{start_batch:02d}.json") if a.rank == 0 else None
    started = time.time()
    for batch in range(start_batch + 1, a.stop_after_batches + 1):
        group = chr(64 + batch)
        local_train = [r for r in rows if r["group"] == group]
        assert len(local_train) == 8 and all(r["split"] == "train" for r in local_train)
        # Read-only held-out gradient, with no optimizer state change.
        optimizer.zero_grad(set_to_none=True)
        held_before = backward(model, head, hidden, diagnostic, 64)
        if a.rank == 0:
            held_gradient = [p.grad.detach().cpu().clone() for p in model.parameters()]
            held_norm2 = sum(float(g.double().square().sum()) for g in held_gradient)
        optimizer.zero_grad(set_to_none=True)
        for update in range(1, LONG_PASS.reuse + 1):
            step = (batch - 1) * LONG_PASS.reuse + update
            optimizer.zero_grad(set_to_none=True)
            value = backward(model, head, hidden, local_train, LONG_PASS.batch_size)
            if a.rank == 0 and update == 1:
                reference = (
                    sum(r["stats"]["all"]["rkl"] for r in previous if r["group"] == group) / LONG_PASS.normalizer
                )
                assert abs(value - reference) < 5e-6, (a.teacher, batch, value, reference)
                raw_dot = torch.zeros((), device="cuda", dtype=torch.float64)
                for p, gh in zip(model.parameters(), held_gradient, strict=False):
                    raw_dot += (p.grad * gh.to("cuda")).sum(dtype=torch.float64)
                before = [p.detach().clone() for p in model.parameters()]
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), LONG_PASS.clip)
            assert torch.isfinite(norm) and math.isfinite(value)
            optimizer.step()
            if a.rank == 0:
                row = dict(
                    step=step,
                    training_batch=batch,
                    group=group,
                    within_batch_update=update,
                    loss=value,
                    grad_norm=float(norm),
                    normalizer=LONG_PASS.normalizer,
                    prompt_ids=[r["id"] for r in bank if r["group"] == group],
                    actual_tokens=sum(len(r["response_ids"]) for r in bank if r["group"] == group),
                    elapsed_segment_seconds=time.time() - started,
                    peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                )
                if update == 1:
                    delta_norm2 = torch.zeros((), device="cuda", dtype=torch.float64)
                    held_dot_delta = delta_norm2.clone()
                    with torch.no_grad():
                        for p, old, gh in zip(model.parameters(), before, held_gradient, strict=False):
                            delta = p - old
                            delta_norm2 += delta.double().square().sum()
                            held_dot_delta += (delta * gh.to("cuda")).sum(dtype=torch.float64)
                    probe = dict(
                        training_batch=batch,
                        group=group,
                        step=step,
                        diagnostic_groups=["M01", "M02", "M03", "M04"],
                        held_before=held_before,
                        held_gradient_norm=held_norm2**0.5,
                        train_gradient_norm=float(norm),
                        raw_train_held_cosine=float(raw_dot) / (float(norm) * held_norm2**0.5),
                        parameter_step_norm=float(delta_norm2.sqrt()),
                        held_dot_actual_update=float(held_dot_delta),
                        descent_alignment=-float(held_dot_delta) / (float(delta_norm2.sqrt()) * held_norm2**0.5),
                        baseline_loss_consistency=value - reference,
                    )
                    row["parameter_step_norm"] = probe["parameter_step_norm"]
                    del before, held_gradient, delta, delta_norm2, held_dot_delta, raw_dot
                history.append(row)
                dump(dest / "training.json", history)
                print(
                    "UPDATE",
                    a.teacher,
                    batch,
                    group,
                    update,
                    "step",
                    step,
                    "KL",
                    round(value, 7),
                    "grad",
                    round(float(norm), 5),
                    flush=True,
                )
            if update == 1:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                local_after = score_local(model, head, hidden, diagnostic, batch, rkl_only=True)
                held_after = all_sum(sum(r["stats"]["all"]["rkl"] for r in local_after) / (64 * LONG_PASS.horizon))
                if a.rank == 0:
                    probe.update(held_after=held_after, finite_held_change=held_after - held_before)
                    probes.append(probe)
                    dump(dest / "update-probes.json", probes)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        previous = collect(a, score_local(model, head, hidden, rows, batch), dest / f"fixed-{batch:02d}.json")
        if batch in [5, 10, 15, 20]:
            fresh(a, model, teacher, rows, batch, dest)
        if batch in [1, 5, 10, 15, 20] or batch == a.stop_after_batches:
            checkpoint(a, model, optimizer, batch, dest)
        if a.rank == 0:
            summary = {
                g: sum(r["stats"]["all"]["rkl"] for r in previous if r["group"] == g) / LONG_PASS.normalizer
                for g in sorted({r["group"] for r in previous})
            }
            dump(
                dest / "progress.json",
                dict(completed_batches=batch, updates=batch * LONG_PASS.reuse, current_group=group, risks=summary),
            )
            print(
                "BATCH_COMPLETE",
                a.teacher,
                batch,
                "held_math",
                sum(summary[f"M{i:02d}"] for i in range(1, 13)) / 12,
                flush=True,
            )
    if a.rank == 0:
        dump(
            dest / ("complete.json" if a.stop_after_batches == 20 else "segment-complete.json"),
            dict(
                completed_batches=a.stop_after_batches,
                updates=LONG_PASS.reuse * a.stop_after_batches,
                finished=a.stop_after_batches == 20,
            ),
        )
    dist.barrier()


def main():
    ap = argparse.ArgumentParser()
    for name in ["models", "out", "cache"]:
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--teacher", choices=["justrl", "polaris"], required=True)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--stop-after-batches", type=int, default=20)
    a = ap.parse_args()
    a.rank = int(os.environ["RANK"])
    a.world = int(os.environ["WORLD_SIZE"])
    assert a.world == 2 and 1 <= a.stop_after_batches <= 20
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    random.seed(LONG_PASS.seed)
    torch.manual_seed(LONG_PASS.seed)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.set_per_process_memory_fraction(0.9)
    a.cap = LONG_PASS.horizon
    a.k = 16
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
    try:
        run(a)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
