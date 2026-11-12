# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Token response, within-trajectory aggregation and component transfer."""

import hashlib
import json
import time
from pathlib import Path

import torch

from .checkpoints import CheckpointStore
from .inputs import source_fingerprint
from .metrics import chunked_gram, tensor_inner
from .reductions import response_denominator
from .response import GradientBank, cotangents, token_signs
from .runtime import checkpoint_identity, collect_memory, gradient_norms, hidden_states, load_model, write_json
from .trajectory import panel, parser, request_cache, tasks


def record_response(
    student, teacher, initial, row, *, cap, probes, seed, token_chunk, bundle, bank, reduction="fixed_cap"
):
    student.zero_grad(set_to_none=True)
    hidden = hidden_states(student, row["prompt_ids"] + row["response_ids"][:-1])
    hidden = hidden[len(row["prompt_ids"]) - 1 :][:cap]
    head = student.lm_head.weight
    key = row["row_id"]
    ht, h0 = [payload["hidden"][key][: len(hidden)].to(hidden.device) for payload in (teacher, initial)]
    signs = token_signs(key, len(hidden), probes, seed, device=hidden.device)
    fields = ["live", "parallel", *[f"sign-{k}" for k in range(probes)]]
    values, norm_values = {}, []
    native = None
    parameters = tuple(student.parameters())
    with gradient_norms(student) as norm:
        for index, (field, dh, dw, moments) in enumerate(
            cotangents(
                hidden,
                head,
                ht,
                teacher["head"],
                (h0, initial["head"]),
                fields=fields,
                signs=signs,
                cap=cap,
                reduction=reduction,
                token_chunk=token_chunk,
                bundle=bundle,
            )
        ):
            student.zero_grad(set_to_none=True)
            norm.zero_()
            torch.autograd.backward((hidden, head), (dh, dw), retain_graph=index < len(fields) - 1)
            norm_values.append(norm.clone())
            if field in ("live", "parallel"):
                bank.add(field)
            if field == "live":
                values["moments"] = moments.cpu().tolist()
                native = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
            elif field == "parallel":
                dot = norm.new_zeros(())
                for reference, p in zip(native, parameters, strict=True):
                    if reference is not None and p.grad is not None:
                        dot.add_(tensor_inner(reference, p.grad))
                values["native_parallel_inner"] = float(dot)
                del native
    values["norm_sq"] = dict(zip(fields, torch.stack(norm_values).cpu().tolist(), strict=True))
    values.update(token_count=len(hidden), denominator=response_denominator(len(hidden), cap, reduction))
    values.update(row_id=key, split=row["split"], group=row.get("diagnostic_group", row.get("group")))
    student.zero_grad(set_to_none=True)
    return values


def run(task, initial_path, root, out, *, probes, seed, token_chunk, bundle, max_groups=None):
    rows, math, groups = panel(task, max_groups)
    signature = dict(
        protocol="trajectory-response-components-v1",
        task=task,
        initial=checkpoint_identity(initial_path),
        student=checkpoint_identity(task["student"]),
        teacher=checkpoint_identity(task["teacher"]),
        records=source_fingerprint([row for _, batch in groups for row in batch]),
        probes=probes,
        seed=seed,
        token_chunk=token_chunk,
        bundle=bundle,
        precision="FP32 master/head, BF16 backbone autocast; FP64 moments/Gram",
    )
    destination = out / f"{task['target']}-s{task['stage']:03d}.json"
    with CheckpointStore(destination.with_suffix(".parts"), signature) as store:
        if destination.exists():
            if json.loads(destination.read_text())["signature"] != signature:
                raise ValueError(f"Stale result {destination}")
            return
        start = time.monotonic()
        teacher, initial = request_cache(task, initial_path, root, rows, math)
        student = load_model(task["student"], checkpointing=task.get("checkpointing", False))
        torch.cuda.reset_peak_memory_stats()
        vectors, measurements, records = [], [], []
        for index, (label, batch) in enumerate(groups):
            tick = time.monotonic()
            bank = GradientBank(student, ("live", "parallel"))
            for row in batch:
                value = record_response(
                    student,
                    teacher,
                    initial,
                    row,
                    cap=task["cap"],
                    reduction=task.get("reduction", "fixed_cap"),
                    probes=probes,
                    seed=seed,
                    token_chunk=token_chunk,
                    bundle=bundle,
                    bank=bank,
                )
                records.append(value)
                store.save_json("row-" + row["row_id"], value)
            for field in ("live", "parallel"):
                vectors.append(bank.export(field, len(batch)).numpy())
                measurements.append(
                    dict(field=field, split=label[0], group=label[1], rows=[r["row_id"] for r in batch])
                )
            del bank
            status = dict(
                task=task["id"],
                group=index + 1,
                groups=len(groups),
                seconds=time.monotonic() - start,
                group_seconds=time.monotonic() - tick,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                resident_bytes=sum(v.nbytes for v in vectors),
            )
            write_json(destination.with_suffix(".progress.json"), status)
            print(json.dumps(dict(event="response_group", **status)), flush=True)
        del student, teacher, initial
        collect_memory()
        tick = time.monotonic()
        gram = chunked_gram(vectors, chunk=16777216, device="cuda").tolist()
        source_hashes = {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in (
                "response_probe.py",
                "response.py",
                "trajectory.py",
                "runtime.py",
                "metrics.py",
                "reductions.py",
            )
        }
        write_json(
            destination,
            dict(
                signature=signature,
                groups=measurements,
                records=records,
                gram=gram,
                seconds=time.monotonic() - start,
                gram_seconds=time.monotonic() - tick,
                source_hashes=source_hashes,
            ),
        )
        print(
            json.dumps(dict(event="response_finished", task=task["id"], seconds=time.monotonic() - start)), flush=True
        )


def main():
    arguments = parser(__doc__)
    arguments.add_argument("--probes", type=int, default=16)
    arguments.add_argument("--seed", type=int, default=20261109)
    arguments.add_argument("--bundle", type=int, default=4)
    args = arguments.parse_args()
    if args.probes < 2 or args.bundle < 1:
        arguments.error("at least two probes and a positive bundle required")
    initial, selected = tasks(args)
    for task in selected:
        run(
            task,
            initial,
            args.manifest.parent,
            args.out,
            probes=args.probes,
            seed=args.seed,
            token_chunk=args.token_chunk,
            bundle=args.bundle,
            max_groups=args.max_groups,
        )


if __name__ == "__main__":
    main()
