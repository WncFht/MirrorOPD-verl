# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Large independent-question panel with exact, streamed B8 gradient Gram.

Directions remain transient on the GPU. Batched prefix VJPs avoid the quadratic
number of individual receiver forwards required by the small logit probe.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .checkpoints import CheckpointStore
from .logit_geometry import direction_inner, logit_pullback
from .logit_probe import progress, site_payload
from .prefixes import branch_tasks, prepare_panel
from .runtime import checkpoint_identity, collect_memory, load_model, prefix_batch_logits, write_json, write_tensors


def grouped_direction(student, selected, indices, payloads, field):
    logits = prefix_batch_logits(student, [selected[i] for i in indices])
    request = torch.stack([payloads[i]["requests"][field] for i in indices]).to(logits.device) / len(indices)
    return logit_pullback(logits, student.parameters(), request)


def stream_group_gram(student, selected, payloads, store):
    groups = {}
    for i, (row, _) in enumerate(selected):
        groups.setdefault((row["split"], row["group"]), []).append(i)
    members = list(groups.values())
    fields = list(payloads[0]["requests"])
    result = {}
    for field_index, field in enumerate(fields):
        n = len(members)
        gram = torch.zeros(n, n, dtype=torch.float64)
        for source, indices in enumerate(members):
            key = f"gram-{field_index:02d}-{source:03d}"
            pairs = store.json(key)
            if pairs is None:
                started = time.monotonic()
                direction = grouped_direction(student, selected, indices, payloads, field)
                pairs = []
                for receiver in range(source, n):
                    if receiver == source:
                        inner = direction_inner(direction, direction)
                    else:
                        other = grouped_direction(student, selected, members[receiver], payloads, field)
                        inner = direction_inner(direction, other)
                        del other
                    pairs.append(inner)
                del direction
                store.save_json(key, pairs)
                progress("group_gram_source_done", field=field, source=source, elapsed=time.monotonic() - started)
            for receiver, inner in enumerate(pairs, start=source):
                gram[source, receiver] = gram[receiver, source] = inner
        result[field] = gram
    return dict(
        groups=[dict(split=key[0], group=key[1], sites=members[i]) for i, key in enumerate(groups)], grams=result
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--stages", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--spectra-per-split", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=24)
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads(args.manifest.read_text())
    tasks, _, selected, requests, identity = prepare_panel(
        manifest,
        args.manifest.parent,
        model="r1",
        out=args.out / "requests",
        per_stratum=160,
        seed=args.seed,
        domains=["math"],
        position_mode="uniform",
    )
    spectrum_indices = set()
    for split in ["train", "heldout"]:
        indices = [i for i, (row, _) in enumerate(selected) if row["split"] == split]
        ordered = sorted(
            indices,
            key=lambda i: hashlib.sha256(f"spectrum:{args.seed}:{selected[i][0]['row_id']}".encode()).hexdigest(),
        )
        spectrum_indices.update(ordered[: args.spectra_per_split])
    for task in branch_tasks(tasks, args.branch):
        if task["stage"] not in args.stages:
            continue
        destination = args.out / f"{args.branch}-s{task['stage']:03d}.json"
        signature = dict(
            protocol="group-logit-v1",
            request=identity,
            checkpoint=checkpoint_identity(task["student"]),
            spectrum_indices=sorted(spectrum_indices),
            iterations=args.iterations,
            attention="math",
            sign="positive-loss-gradient",
            grouping="original-bank-B8",
            batching="right-padded-prefixes",
        )
        with CheckpointStore(destination.with_suffix(".parts"), signature) as store:
            if destination.exists():
                if json.loads(destination.read_text())["signature"] != json.loads(json.dumps(signature)):
                    raise ValueError(f"Stale group probe: {destination}")
                continue
            targets = requests if task["stage"] == 0 else {task["target"]: requests[task["target"]]}
            student = load_model(task["student"])
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            payloads = []
            with sdpa_kernel(SDPBackend.MATH):
                for i, (row, pos) in enumerate(selected):
                    key = f"site-{i:03d}"
                    payload = store.tensors(key)
                    if payload is None:
                        payload = site_payload(
                            student,
                            row,
                            pos,
                            targets,
                            i,
                            iterations=args.iterations if i in spectrum_indices else 0,
                            seed=args.seed + i,
                        )
                        store.save_tensors(key, payload)
                    payloads.append(payload)
                    progress("large_site_done", stage=task["stage"], site=i, total=len(selected))
                group_result = stream_group_gram(student, selected, payloads, store)
            write_tensors(
                destination.with_suffix(".pt"),
                dict(
                    signature=signature,
                    **group_result,
                    spectra={i: p["spectrum"] for i, p in enumerate(payloads) if p["spectrum"]},
                ),
            )
            write_json(
                destination,
                dict(
                    signature=signature,
                    task=task,
                    sites=[p["record"] for p in payloads],
                    execution=dict(
                        elapsed_seconds=time.monotonic() - started,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    ),
                ),
            )
            del student, payloads
            collect_memory()


if __name__ == "__main__":
    main()
