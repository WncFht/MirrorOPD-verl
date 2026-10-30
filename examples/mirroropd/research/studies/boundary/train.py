# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Full-parameter fixed-state RKL, explicit 64-update TB arms and output caches."""

import argparse
import gc
import math
import os
import random
import time

import torch
from transformers import AutoTokenizer

from examples.mirroropd.assets.paths import checkpoint_path
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.core.metrics import fixed_metrics
from examples.mirroropd.research.studies.boundary.common import ARMS, CACHE, GROUPS, MODELS, OUT, model_path
from examples.mirroropd.research.studies.boundary.native import bank, hidden, load, setup


def load_teacher(teacher, student, rows, device):
    directory = CACHE / "teachers" / teacher / student
    receipt = read(directory / "complete.json")
    assert receipt["teacher"] == list(MODELS[teacher])
    head = torch.load(directory / "head.pt", map_location=device, weights_only=True).float()
    states = {}
    for row in rows:
        payload = torch.load(directory / f"{row['id']:04d}.pt", map_location="cpu", weights_only=True)
        assert payload["prompt_ids"] == row["prompt_ids"] and payload["response_ids"] == row["response_ids"]
        assert tuple(payload["teacher"]) == MODELS[teacher]
        states[row["id"]] = payload["hidden"]
    return head, states


def backward(model, head, states, rows):
    value = 0.0
    device = model.lm_head.weight.device
    for row in rows:
        hp = hidden(model, row).to(device)
        leaf = hp.detach().float().requires_grad_(True)
        hq = states[row["id"]].to(device).float()
        for start in range(0, len(hp), 64):
            lp = torch.nn.functional.linear(leaf[start : start + 64], model.lm_head.weight).log_softmax(-1)
            with torch.no_grad():
                lq = torch.nn.functional.linear(hq[start : start + 64], head).log_softmax(-1)[:, : lp.shape[-1]]
            loss = (lp.exp() * (lp - lq)).sum() / (16 * 1024)
            value += float(loss.detach())
            loss.backward()
        hp.backward(leaf.grad.to(hp.dtype))
        del hp, leaf, hq, lp, lq, loss
    return value


@torch.no_grad()
def score(model, head, states, rows, step):
    results = []
    device = model.lm_head.weight.device
    bands_spec = [("all", 0, 1024), ("early", 0, 256), ("middle", 256, 512), ("late", 512, 1024)]
    for index, row in enumerate(rows, 1):
        hp = hidden(model, row).to(device).float()
        hq = states[row["id"]].to(device).float()
        bands = {name: dict(tokens=max(0, min(len(hp), hi) - lo)) for name, lo, hi in bands_spec}
        for start in range(0, len(hp), 64):
            sp = torch.nn.functional.linear(hp[start : start + 64], model.lm_head.weight)
            tq = torch.nn.functional.linear(hq[start : start + 64], head)
            values = fixed_metrics(sp, tq)
            values["extra_teacher_mass"] = tq.softmax(-1)[:, sp.shape[-1] :].sum(-1)
            for name, lo, hi in bands_spec:
                left, right = max(lo - start, 0), min(hi - start, len(sp))
                if right > left:
                    for key, tensor in values.items():
                        bands[name][key] = bands[name].get(key, 0.0) + float(
                            tensor[left:right].sum(dtype=torch.float64)
                        )
        results.append(
            dict(
                id=row["id"],
                group=row["group"],
                domain=row["domain"],
                split=row["split"],
                step=step,
                length=len(hp),
                stats=bands,
                source=row["source"],
            )
        )
        if index % 64 == 0:
            print("SCORE", step, index, len(rows), flush=True)
    return results


def checkpoint(model, optimizer, student, arm, step, rows):
    dest = checkpoint_path(CACHE, "checkpoints") / arm / f"step{step:03d}"
    dest.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.save_pretrained(dest, state_dict=state, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(model_path(student)).save_pretrained(dest)
    del state
    optim = optimizer.state_dict()
    cpu = {
        "param_groups": optim["param_groups"],
        "state": {
            k: {n: (v.detach().cpu() if torch.is_tensor(v) else v) for n, v in s.items()}
            for k, s in optim["state"].items()
        },
    }
    torch.save(
        dict(
            optimizer=cpu,
            step=step,
            arm=arm,
            models=ARMS[arm],
            model_revisions=MODELS,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate(),
            trajectories=[{k: r[k] for k in ["id", "prompt_ids", "response_ids", "source", "seed"]} for r in rows],
        ),
        dest / "optimizer.pt",
    )
    dump(
        dest / "complete.json", dict(step=step, optimizer_steps=sorted({int(s["step"]) for s in cpu["state"].values()}))
    )
    print("CHECKPOINT", arm, step, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    setup()
    random.seed(42)
    torch.manual_seed(42)
    student, teacher = ARMS[args.arm]
    dest = OUT / ("smoke" if args.smoke else "arms") / args.arm
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "complete.json").exists():
        return
    if (dest / "training.json").exists():
        raise RuntimeError("Partial run exists; preserve it and explicitly restart in a distinct attempt")
    rows = [r for r in bank(student) if r["group"] in GROUPS]
    assert len(rows) == 256
    parallel = student == "r7"
    model = load(student, student=True, parallel=parallel)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    head, states = load_teacher(teacher, student, rows, model.lm_head.weight.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False
    )
    protocol = read(OUT / "protocol.json") | dict(
        arm=args.arm,
        student=student,
        teacher=teacher,
        parameters=sum(p.numel() for p in model.parameters()),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        parallelism="two-GPU layer model parallel" if parallel else "one-GPU gradient accumulation",
        parallelism_reason="FP32 7B state exceeds one H800; global batch16 and loss denominator are preserved",
        smoke=args.smoke,
    )
    dump(dest / "protocol.json", protocol)
    panel = rows[:16] if args.smoke else rows
    dump(dest / "fixed-000.json", score(model, head, states, panel, 0))
    history = []
    started = time.time()
    for step in range(1, args.steps + 1):
        group = chr(65 + (step - 1) // 16)
        train_rows = [r for r in rows if r["group"] == group]
        assert len(train_rows) == 16 and all(r["split"] == "train" for r in train_rows)
        optimizer.zero_grad(set_to_none=True)
        update_started = time.time()
        loss = backward(model, head, states, train_rows)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
        assert math.isfinite(loss) and torch.isfinite(norm)
        backward_finished = time.time()
        before = [
            p.detach().cpu() if parallel and p.numel() > 200_000_000 else p.detach().clone() for p in model.parameters()
        ]
        snapshot_finished = time.time()
        optimizer.step()
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)
        optimizer_finished = time.time()
        delta = 0.0
        with torch.no_grad():
            for p, old in zip(model.parameters(), before, strict=True):
                if old.device.type == "cpu":
                    old_flat, new_flat = old.reshape(-1), p.detach().reshape(-1)
                    for start in range(0, p.numel(), 8 * 1024 * 1024):
                        stop = start + 8 * 1024 * 1024
                        part = old_flat[start:stop].to(p.device)
                        part.sub_(new_flat[start:stop]).square_()
                        delta += float(part.sum(dtype=torch.float64))
                    del part, old_flat, new_flat
                else:
                    old.sub_(p.detach())
                    delta += float(old.square_().sum(dtype=torch.float64))
        del before, old
        record = dict(
            step=step,
            group=group,
            loss=loss,
            grad_norm=float(norm),
            parameter_step_norm=delta**0.5,
            actual_tokens=sum(len(r["response_ids"]) for r in train_rows),
            elapsed=time.time() - started,
            peak_gib=[torch.cuda.max_memory_allocated(d) / 2**30 for d in range(torch.cuda.device_count())],
            backward_seconds=backward_finished - update_started,
            snapshot_seconds=snapshot_finished - backward_finished,
            optimizer_seconds=optimizer_finished - snapshot_finished,
            displacement_measure_seconds=time.time() - optimizer_finished,
            displacement_measurement="GPU snapshots; large CPU snapshots measured in GPU chunks",
        )
        history.append(record)
        dump(dest / "training.json", history)
        print("UPDATE", args.arm, record, flush=True)
        optimizer.zero_grad(set_to_none=True)
        if step % 16 == 0 or step == args.steps:
            dump(dest / f"fixed-{step:03d}.json", score(model, head, states, panel, step))
        if step in [32, 64] and not args.smoke:
            checkpoint(model, optimizer, student, args.arm, step, rows)
        gc.collect()
    dump(
        dest / "complete.json", dict(updates=args.steps, elapsed=time.time() - started, arm=args.arm, smoke=args.smoke)
    )


if __name__ == "__main__":
    main()
