# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Restored-history AdamW interventions with exact native heldout evaluation."""

import hashlib
import json
import time
from pathlib import Path

import torch

from .checkpoints import CheckpointStore
from .inputs import source_fingerprint, validate_source_exposure
from .metrics import squared_norm, tensor_inner
from .runtime import checkpoint_identity, collect_memory, flatten_tensors, load_model, write_json
from .trajectory import panel, parser, request_cache, select_source_groups, tasks
from .update_probe import evaluate, restore_step, source_directions


def restored_step(model, gradient, state, *, clip=1.0):
    """One PyTorch AdamW step, preserving saved hyperparameters and moments."""
    parameters = tuple(model.parameters())
    ids = [i for group in state["param_groups"] for i in group["params"]]
    if len(ids) != len(parameters):
        raise ValueError("Optimizer parameter count differs from model")
    for i, p in zip(ids, parameters, strict=True):
        if state["state"][i]["exp_avg"].shape != p.shape:
            raise ValueError("Optimizer parameter order/shape differs from model")
    before = flatten_tensors(parameters, [p.detach() for p in parameters])
    optimizer = torch.optim.AdamW(parameters)
    optimizer.load_state_dict(state)
    old_steps = {int(v["step"].item()) for v in optimizer.state.values()}
    if len(old_steps) != 1:
        raise ValueError("Mixed optimizer history steps")
    offset = 0
    for p in parameters:
        size = p.numel()
        p.grad = gradient[offset : offset + size].reshape_as(p).to(p.device)
        offset += size
    if offset != gradient.numel():
        raise ValueError("Gradient size differs from model")
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, clip))
    optimizer.step()
    new_steps = {int(v["step"].item()) for v in optimizer.state.values()}
    if new_steps != {next(iter(old_steps)) + 1}:
        raise ValueError("Adam did not advance exactly one saved step")
    after = flatten_tensors(parameters, [p.detach() for p in parameters])
    optimizer.zero_grad(set_to_none=True)
    return before - after, dict(gradient_norm=norm, step_before=next(iter(old_steps)), step_after=next(iter(new_steps)))


def run(task, initial_path, root, out, *, token_chunk, scales, source_indices=(15, 17, 19)):
    optimizer_path = Path(task["student"]) / "optimizer.pt"
    if not optimizer_path.exists():
        raise FileNotFoundError(optimizer_path)
    rows, math, groups = panel(task)
    selected = select_source_groups(groups, source_indices)
    held = [(label, batch) for label, batch in groups if label[0] == "heldout"]
    validate_source_exposure(task, [r for _, batch in selected for r in batch])
    calibration = {label[1] for label, _ in held[: len(held) // 2]}
    stat = optimizer_path.stat()
    signature = dict(
        protocol="restored-adam-native-v1",
        task=task,
        student=checkpoint_identity(task["student"]),
        teacher=checkpoint_identity(task["teacher"]),
        records=source_fingerprint(math),
        optimizer=dict(path=str(optimizer_path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns),
        scales=scales,
        clip=1.0,
        source_indices=list(source_indices),
        calibration=sorted(calibration),
        token_chunk=token_chunk,
        precision="FP32 master/head/gradient; BF16 backbone autocast",
    )
    destination = out / f"{task['target']}-s{task['stage']:03d}.adam.json"
    with CheckpointStore(destination.with_suffix(".parts"), signature) as store:
        if destination.exists():
            if json.loads(destination.read_text())["signature"] != signature:
                raise ValueError(f"Stale Adam result: {destination}")
            return
        started = time.monotonic()
        teacher, initial = request_cache(task, initial_path, root, rows, math)
        model = load_model(task["student"])
        parameters = tuple(model.parameters())
        baseline = flatten_tensors(parameters, [p.detach() for p in parameters])
        original = store.json("baseline")
        if original is None:
            original = evaluate(
                model,
                teacher,
                math,
                cap=task["cap"],
                reduction=task.get("reduction", "fixed_cap"),
                token_chunk=token_chunk,
            )
            store.save_json("baseline", original)
        receiver = {
            label[1]: source_directions(
                model,
                teacher,
                initial,
                batch,
                cap=task["cap"],
                reduction=task.get("reduction", "fixed_cap"),
                token_chunk=token_chunk,
                fields=("live",),
            )["native"]
            for label, batch in held
        }
        trials = []
        for label, batch in selected:
            source_name = label[1]
            keys = [f"{source_name}-{scale}" for scale in scales]
            saved = [store.json(key) for key in keys]
            if all(value is not None for value in saved):
                trials.extend(saved)
                continue
            gradient = source_directions(
                model,
                teacher,
                initial,
                batch,
                cap=task["cap"],
                reduction=task.get("reduction", "fixed_cap"),
                token_chunk=token_chunk,
                fields=("live",),
            )["native"]
            state = torch.load(optimizer_path, map_location="cpu", weights_only=True, mmap=True)
            if state["step"] != task["stage"]:
                raise ValueError("Optimizer and model stage mismatch")
            delta, metadata = restored_step(model, gradient, state["optimizer"])
            restore_step(model, baseline, None, 0)
            predictions = {name: float(tensor_inner(g, delta)) for name, g in receiver.items()}
            predictions[source_name] = float(tensor_inner(gradient, delta))
            metadata.update(
                displacement=squared_norm(delta) ** 0.5,
                gradient_step_cosine=float(tensor_inner(gradient, delta))
                / (squared_norm(gradient) * squared_norm(delta)) ** 0.5,
            )
            del state, gradient
            collect_memory()
            device_delta = delta.cuda()
            evaluation = [r for _, h in held for r in h] + batch
            for scale, key, previous in zip(scales, keys, saved, strict=True):
                if previous is None:
                    restore_step(model, baseline, device_delta, scale)
                    values = evaluate(
                        model,
                        teacher,
                        evaluation,
                        cap=task["cap"],
                        reduction=task.get("reduction", "fixed_cap"),
                        token_chunk=token_chunk,
                    )
                    restore_step(model, baseline, None, 0)
                    previous = dict(
                        source=source_name,
                        scale=scale,
                        records=values,
                        predictions={name: scale * v for name, v in predictions.items()},
                        **metadata,
                    )
                    store.save_json(key, previous)
                trials.append(previous)
                print(
                    json.dumps(
                        dict(
                            event="restored_adam_trial",
                            branch=task["target"],
                            source=source_name,
                            scale=scale,
                            seconds=time.monotonic() - started,
                        )
                    ),
                    flush=True,
                )
            del device_delta, delta
        write_json(
            destination,
            dict(
                signature=signature,
                baseline=original,
                trials=trials,
                seconds=time.monotonic() - started,
                source_hashes={
                    name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                    for name in ("adam_probe.py", "update_probe.py", "response.py", "runtime.py")
                },
            ),
        )


def main():
    arguments = parser(__doc__)
    arguments.add_argument("--source-indices", type=int, nargs="+", default=[15, 17, 19])
    arguments.add_argument("--scales", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    args = arguments.parse_args()
    if args.max_groups is not None or any(scale <= 0 for scale in args.scales):
        arguments.error("complete panel and positive displacement scales required")
    initial, selected = tasks(args)
    for task in selected:
        run(
            task,
            initial,
            args.manifest.parent,
            args.out,
            token_chunk=args.token_chunk,
            scales=args.scales,
            source_indices=args.source_indices,
        )


if __name__ == "__main__":
    main()
