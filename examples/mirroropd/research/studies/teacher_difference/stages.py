# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Durable fixed-bank stage measurement, with exact batch Gram sufficient statistics."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .inputs import load_records, normalize_records, source_fingerprint, validate_records
from .metrics import chunked_gram, moment_summary, summarize_gram
from .pac import compute_batch
from .runtime import checkpoint_identity, collect_memory, load_model, teacher_cache, write_json


def disk_gram(paths, chunk=16777216, *, resident_vectors=None):
    vectors = resident_vectors if resident_vectors is not None else [np.load(p, mmap_mode="r") for p in paths]

    def progress(start, total):
        if start // chunk % 8 == 0:
            print(json.dumps(dict(event="gram_progress", parameters=start, total=total)), flush=True)

    return chunked_gram(vectors, chunk, device="cuda", progress=progress).numpy()


def subset_summary(groups, gram, indices):
    selected = [groups[i] for i in indices]
    count = sum(len(g["records"]) for g in selected)
    weights = np.array([len(g["records"]) / count for g in selected])
    sub = gram[np.ix_(indices, indices)]
    rows = sum((g["records"] for g in selected), [])
    out = moment_summary(
        pressure=np.mean([r["pressure"] for r in rows]),
        second=np.mean([r["gradient_norm_sq"] for r in rows]),
        mean_norm=weights @ sub @ weights,
        samples=count,
    )
    out["loss_mean"] = float(np.mean([r["loss"] for r in rows]))
    out["geometry"] = summarize_gram(sub)
    if len(selected) > 1:
        off = (sub.sum() - np.trace(sub)) / (len(selected) * (len(selected) - 1))
        out["cross_batch_inner"] = float(off)
        out["cross_batch_normalized"] = float(off / np.diag(sub).mean())
    return out


def run(task, root):
    out = root / "results" / task["id"]
    rows = normalize_records(load_records(task["records"]), cap=task["cap"], bank_id=task["bank"])
    fingerprint = source_fingerprint(rows)
    expected = task.get("input_identity")
    if expected is not None:
        actual = dict(
            student=checkpoint_identity(task["student"]),
            teacher=checkpoint_identity(task["teacher"]),
            records=fingerprint,
        )
        if actual != expected:
            raise ValueError(f"Inputs changed after plan compilation: {task['id']}")
    signature = dict(task=task, fingerprint=fingerprint, schema=2)
    if (out / "result.json").exists():
        previous = json.loads((out / "result.json").read_text())
        if previous["signature"] != signature:
            raise ValueError(f"Stale completed result: {out}")
        return
    if (out / "input.json").exists() and json.loads((out / "input.json").read_text()) != signature:
        raise ValueError(f"Partial results use different inputs: {out}")
    write_json(out / "input.json", signature)
    print(json.dumps(dict(event="start", task=task["id"], audit=validate_records(rows))), flush=True)
    device = torch.device("cuda")
    cached = teacher_cache(root / "cache", task["teacher"], rows, device=device)
    hidden, head = cached["hidden"], cached["head"].to(device)
    del cached
    student = load_model(task["student"], checkpointing=task.get("checkpointing", False), device=device)
    groups = []
    # Contiguous groups never cross dataset/domain/split boundaries.
    group_rows = []
    for row in rows:
        label = (
            row.get("split"),
            row.get("domain"),
            row.get("diagnostic_group", row.get("group", row.get("domain", "all"))),
        )
        if not group_rows or group_rows[-1][0] != label or len(group_rows[-1][1]) >= task["group_size"]:
            group_rows.append((label, []))
        group_rows[-1][1].append(row)
    started = time.monotonic()
    # Optional on hosts with sufficient RAM: keep the already computed CPU
    # vectors while still writing resumable copies. Avoid repeatedly reading
    # hundreds of GB through interleaved network-backed mmap windows.
    resident_vectors = [] if os.environ.get("TEACHER_DIFFERENCE_RESIDENT_GRAM") == "1" else None
    for i, (label, batch) in enumerate(group_rows):
        path = out / f"group-{i:03d}.json"
        vector_path = out / f"group-{i:03d}.npy"
        if path.exists() and vector_path.exists():
            group = json.loads(path.read_text())
            if resident_vectors is not None:
                resident_vectors.append(np.load(vector_path))
        else:
            result = compute_batch(
                student,
                head,
                hidden,
                batch,
                cap=task["cap"],
                reduction=task.get("reduction", "fixed_cap"),
                retain_samples=False,
            )
            vector = result["gradient_vectors"][0].numpy()
            np.save(vector_path, vector)
            if resident_vectors is not None:
                resident_vectors.append(vector)
            records = [
                dict(
                    row_id=r["row_id"],
                    domain=r.get("domain"),
                    split=r.get("split"),
                    pressure=float(result["pressure"][j]),
                    loss=result["losses"][j],
                    token_count=result["token_counts"][j],
                    denominator=result["denominators"][j],
                    gradient_norm_sq=float(result["gradient_norm_sq"][j]),
                )
                for j, r in enumerate(batch)
            ]
            group = dict(label=label[-1], domain=batch[0].get("domain"), split=batch[0].get("split"), records=records)
            write_json(path, group)
            del result, vector
            student.zero_grad(set_to_none=True)
        groups.append(group)
        print(
            json.dumps(
                dict(
                    event="group_done",
                    task=task["id"],
                    group=i,
                    groups=len(group_rows),
                    elapsed=time.monotonic() - started,
                )
            ),
            flush=True,
        )
    del student, head, hidden
    collect_memory()
    gram = disk_gram([out / f"group-{i:03d}.npy" for i in range(len(groups))], resident_vectors=resident_vectors)
    subsets = {"all": list(range(len(groups)))}
    for i, g in enumerate(groups):
        subsets.setdefault(f"{g['split']}/{g['domain']}", []).append(i)
    summaries = {key: subset_summary(groups, gram, ix) for key, ix in subsets.items()}
    output = dict(
        finished=True,
        signature=signature,
        audit=validate_records(rows),
        summaries=summaries,
        groups=groups,
        geometry=summarize_gram(gram),
        units="P: fixed-cap position average; A/C: trajectory gradient moments; gradients: positive loss derivative",
        precision="FP32 master/head and gradient, BF16 backbone autocast; native full-vocab RKL recompute",
    )
    write_json(out / "result.json", output)
    # The small exact Gram and per-record moments are sufficient for all subset summaries.
    for p in out.glob("group-*.npy"):
        p.unlink()
    print(
        json.dumps(
            dict(
                event="finished",
                task=task["id"],
                summary={k: v for k, v in summaries["all"].items() if k != "geometry"},
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lane", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("execution", {}).get("resident_gram", False):
        os.environ["TEACHER_DIFFERENCE_RESIDENT_GRAM"] = "1"
    paths = {t[key] for t in manifest["tasks"] if t["lane"] == args.lane for key in ["student", "teacher"]}
    for path in sorted(paths):
        checkpoint_identity(path)
    for task in manifest["tasks"]:
        if task["lane"] == args.lane:
            run(task, args.manifest.parent)


if __name__ == "__main__":
    main()
