# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Full-trajectory live/frozen cross-field Gram with linear model work.

Group means stay in CPU RAM; one bounded CUDA contraction computes every pair.
Only small statistics are published. Interrupted tasks recompute transient
vectors, rather than treating an incomplete cross-field Gram as resumable.
"""

import hashlib
import json
import time
from pathlib import Path

import torch

from .checkpoints import CheckpointStore
from .inputs import source_fingerprint
from .metrics import chunked_gram
from .pac import compute_batch
from .runtime import checkpoint_identity, collect_memory, load_model, write_json
from .trajectory import panel, parser, request_cache, tasks


def run(task, initial_path, root, out, *, max_groups=None, token_chunk=128):
    rows, math, groups = panel(task, max_groups)
    selected = [r for _, batch in groups for r in batch]
    signature = dict(
        protocol="trajectory-live-frozen-v1",
        task=task,
        initial=checkpoint_identity(initial_path),
        student=checkpoint_identity(task["student"]),
        teacher=checkpoint_identity(task["teacher"]),
        records=source_fingerprint(selected),
        cap=task["cap"],
        token_chunk=token_chunk,
        parameter_order="model.parameters",
        precision="FP32 master/head, BF16 backbone autocast",
        gram="FP64 CUDA block contraction; FP32 CPU resident group means",
    )
    destination = out / f"{task['target']}-s{task['stage']:03d}.json"
    with CheckpointStore(destination.with_suffix(".parts"), signature):
        if destination.exists():
            if json.loads(destination.read_text())["signature"] != signature:
                raise ValueError(f"Stale result: {destination}")
            return
        started = time.monotonic()
        source_hashes = {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("trajectory_probe.py", "requests.py", "pac.py", "runtime.py", "metrics.py")
        }
        teacher, initial = request_cache(task, initial_path, root, rows, math)
        student = load_model(task["student"], checkpointing=task.get("checkpointing", False))
        torch.cuda.reset_peak_memory_stats()
        vectors, measurements = [], []
        for field in ("live", "frozen"):
            for index, (label, batch) in enumerate(groups):
                tick = time.monotonic()
                result = compute_batch(
                    student,
                    teacher["head"],
                    teacher["hidden"],
                    batch,
                    cap=task["cap"],
                    retain_samples=False,
                    initial=initial if field == "frozen" else None,
                    token_chunk=token_chunk,
                )
                vectors.append(result.pop("gradient_vectors")[0].numpy())
                measurements.append(
                    dict(
                        field=field,
                        split=label[0],
                        group=label[1],
                        rows=[r["row_id"] for r in batch],
                        losses=result["losses"],
                        pressure=result["pressure"].tolist(),
                        second=result["gradient_norm_sq"].tolist(),
                        request=result["request_diagnostics"],
                    )
                )
                student.zero_grad(set_to_none=True)
                write_json(
                    destination.with_suffix(".progress.json"),
                    dict(
                        field=field,
                        group=index,
                        total_groups=len(groups),
                        seconds=time.monotonic() - started,
                        last_group_seconds=time.monotonic() - tick,
                        resident_bytes=sum(v.nbytes for v in vectors),
                        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                    ),
                )
                print(
                    json.dumps(
                        dict(
                            event="trajectory_group",
                            task=task["id"],
                            field=field,
                            group=index,
                            seconds=time.monotonic() - tick,
                        )
                    ),
                    flush=True,
                )
        del student, teacher, initial
        collect_memory()
        gram_started = time.monotonic()

        def progress(start, total):
            print(json.dumps(dict(event="trajectory_gram", task=task["id"], parameters=start, total=total)), flush=True)

        gram = chunked_gram(vectors, chunk=16777216, device="cuda", progress=progress).tolist()
        write_json(
            destination,
            dict(
                signature=signature,
                source_hashes=source_hashes,
                groups=measurements,
                gram=gram,
                seconds=time.monotonic() - started,
                gram_seconds=time.monotonic() - gram_started,
            ),
        )
        print(
            json.dumps(dict(event="trajectory_finished", task=task["id"], seconds=time.monotonic() - started)),
            flush=True,
        )


def main():
    args = parser(__doc__).parse_args()
    initial_path, selected = tasks(args)
    for task in selected:
        run(
            task,
            initial_path,
            args.manifest.parent,
            args.out,
            max_groups=args.max_groups,
            token_chunk=args.token_chunk,
        )


if __name__ == "__main__":
    main()
