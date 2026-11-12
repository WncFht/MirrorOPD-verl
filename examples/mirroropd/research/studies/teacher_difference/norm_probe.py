# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Resumable frozen-request token moments without parameter vector storage."""

import json
import time

import torch

from .checkpoints import CheckpointStore
from .inputs import source_fingerprint
from .reductions import response_denominator
from .response import cotangents, token_signs
from .runtime import checkpoint_identity, gradient_norms, hidden_states, load_model, write_json
from .trajectory import panel, parser, request_cache, tasks


def run(task, initial_path, root, out, *, probes, seed, token_chunk, bundle):
    rows, math, groups = panel(task)
    signature = dict(
        protocol="frozen-token-moments-v1",
        task=task,
        student=checkpoint_identity(task["student"]),
        teacher=checkpoint_identity(task["teacher"]),
        initial=checkpoint_identity(initial_path),
        records=source_fingerprint(math),
        probes=probes,
        seed=seed,
        token_chunk=token_chunk,
        bundle=bundle,
    )
    destination = out / f"{task['target']}-s{task['stage']:03d}.norms.json"
    with CheckpointStore(destination.with_suffix(".parts"), signature) as store:
        if destination.exists():
            if json.loads(destination.read_text())["signature"] != signature:
                raise ValueError(f"Stale result {destination}")
            return
        teacher, initial = request_cache(task, initial_path, root, rows, math)
        model = load_model(task["student"])
        fields = ["frozen", *[f"frozen-sign-{k}" for k in range(probes)]]
        records = []
        started = time.monotonic()
        for index, (label, batch) in enumerate(groups):
            for row in batch:
                key = row["row_id"]
                value = store.json("row-" + key)
                if value is None:
                    model.zero_grad(set_to_none=True)
                    hs = hidden_states(model, row["prompt_ids"] + row["response_ids"][:-1])[
                        len(row["prompt_ids"]) - 1 :
                    ][: task["cap"]]
                    ht, h0 = [payload["hidden"][key][: len(hs)].to(hs.device) for payload in (teacher, initial)]
                    signs = token_signs(key, len(hs), probes, seed, device=hs.device)
                    values, moments = [], None
                    with gradient_norms(model) as norm:
                        for j, (_, dh, dw, info) in enumerate(
                            cotangents(
                                hs,
                                model.lm_head.weight,
                                ht,
                                teacher["head"],
                                (h0, initial["head"]),
                                fields=fields,
                                cap=task["cap"],
                                reduction=task.get("reduction", "fixed_cap"),
                                token_chunk=token_chunk,
                                bundle=bundle,
                                signs=signs,
                            )
                        ):
                            model.zero_grad(set_to_none=True)
                            norm.zero_()
                            torch.autograd.backward(
                                (hs, model.lm_head.weight), (dh, dw), retain_graph=j < len(fields) - 1
                            )
                            values.append(norm.clone())
                            if info is not None:
                                moments = info.cpu().tolist()
                    value = dict(
                        row_id=key,
                        token_count=len(hs),
                        denominator=response_denominator(len(hs), task["cap"], task.get("reduction", "fixed_cap")),
                        group=label[1],
                        split=label[0],
                        moments=moments,
                        norm_sq=dict(zip(fields, torch.stack(values).cpu().tolist(), strict=True)),
                    )
                    store.save_json("row-" + key, value)
                records.append(value)
            status = dict(
                group=index + 1,
                groups=len(groups),
                seconds=time.monotonic() - started,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            )
            write_json(destination.with_suffix(".progress.json"), status)
            print(json.dumps(dict(event="frozen_norm_group", branch=task["target"], **status)), flush=True)
        write_json(destination, dict(signature=signature, records=records, seconds=time.monotonic() - started))


def main():
    arguments = parser(__doc__)
    arguments.add_argument("--probes", type=int, default=8)
    arguments.add_argument("--seed", type=int, default=20261109)
    arguments.add_argument("--bundle", type=int, default=4)
    args = arguments.parse_args()
    if args.max_groups is not None or args.probes < 2 or args.bundle < 1:
        arguments.error("complete panel, at least two probes and positive bundle required")
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
        )


if __name__ == "__main__":
    main()
