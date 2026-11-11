# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""All-layer ISO transport and stage subspace geometry; frames never written."""

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path

import torch

from examples.mirroropd.research.core.io import dump, read

from ..absorption.protocol import RunContext
from ..scan import thin_svd
from ..weights import DEFAULT_SCOPE, TensorCheckpoint, tensor_digest
from .math import subspaces, transport


def run(manifest, output, stop, shards=1, shard_index=0, fractions=(0.5, 0.75, 0.9, 0.95, 0.99)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    context = RunContext(output, stop_paths=(Path(stop),))
    jobs = read(manifest)["jobs"]
    groups = defaultdict(list)
    for job in jobs:
        groups[job["base"]].append(job)
    stores = {}
    started = time.time()
    completed = 0

    def frame(path, key):
        context.check_stop()
        if path not in stores:
            stores[path] = TensorCheckpoint(path)
        raw = stores[path].get(key)
        result = thin_svd(raw.cuda().double(), driver="gesvd", transpose_wide=True)
        return result, tensor_digest(raw)

    for base, paired in groups.items():
        stores.setdefault(base, TensorCheckpoint(base))
        for index, key in enumerate(stores[base].selected(DEFAULT_SCOPE)):
            if index % shards != shard_index:
                continue
            f0, h0 = frame(base, key)
            rows = []
            for job in paired:
                ft, ht = frame(job["teacher_path"], key)

                def compare(incoming, outgoing, transition, hashes, rows=rows, job=job):
                    rows.append(
                        dict(
                            job_id=job["job_id"],
                            teacher=job["teacher"],
                            transition=transition,
                            input_sha256=hashes,
                            transport=[transport(incoming, outgoing, f) for f in fractions],
                            subspaces=subspaces(incoming, outgoing),
                        )
                    )

                compare(f0, ft, "base-to-teacher", [h0, ht])
                points = {"endpoint": job["endpoint"], **job["gradient_stages"]}

                def step(item, job=job):
                    label, _ = item
                    return (
                        0
                        if label == "initial"
                        else job["step"]
                        if label == "endpoint"
                        else int(re.search(r"\d+", label)[0])
                    )

                previous, hp, label0 = f0, h0, "initial"
                seen = {base}
                for label, path in sorted(points.items(), key=step):
                    if path in seen:
                        continue
                    seen.add(path)
                    fs, hs = frame(path, key)
                    compare(f0, fs, f"base-to-{label}", [h0, hs])
                    compare(ft, fs, f"teacher-to-{label}", [ht, hs])
                    if label0 != "initial":
                        compare(previous, fs, f"{label0}-to-{label}", [hp, hs])
                    previous, hp, label0 = fs, hs, label
                del ft, previous
            dump(
                output / f"{completed:05d}.json",
                dict(tensor=key, base=base, shape=list(stores[base].shapes[key]), rows=rows),
            )
            completed += 1
            dump(output / "progress.json", dict(completed=completed, tensor=key, wall_seconds=time.time() - started))
            del f0, rows
    dump(output / "complete.json", dict(completed=completed, wall_seconds=time.time() - started))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stop", required=True)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    args = vars(p.parse_args())
    torch.set_num_threads(2)
    run(**args)


if __name__ == "__main__":
    main()
