# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed-prefix native-request, local-spectrum and cross-logit diagnostics."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .checkpoints import CheckpointStore
from .logit_geometry import (
    LogitJacobian,
    direction_inner,
    leading_spectrum,
    logit_pullback,
    request_projection,
    response_summary,
)
from .prefixes import branch_tasks, prepare_panel
from .runtime import checkpoint_identity, collect_memory, load_model, prefix_logits, write_json, write_tensors
from .signals import profile_and_signal


def progress(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def measure_site(student, row, position, fields, *, iterations, seed):
    """One graph for local requests and Krylov products, no parameter export."""
    started = time.monotonic()
    operator = LogitJacobian(prefix_logits(student, row, position), student.parameters())
    spectrum = (
        leading_spectrum(operator.kernel, operator.logits, iterations=iterations, seed=seed) if iterations else {}
    )
    results = {}
    for name, request in fields.items():
        b = request.to(operator.logits.device)
        direction = operator.pullback(b)
        energy = direction_inner(direction, direction)
        if spectrum:
            response = operator.pushforward(direction)
            results[name] = dict(
                **response_summary(b, response), gradient_norm_sq=energy, **request_projection(b, spectrum)
            )
            del response
        else:
            pressure = float(b.double().square().sum())
            results[name] = dict(pressure=pressure, gradient_norm_sq=energy, rayleigh=energy / max(pressure, 1e-30))
        del direction
    return dict(fields=results, spectrum=spectrum, elapsed_seconds=time.monotonic() - started)


def site_payload(student, row, position, requests, index, *, iterations, seed):
    with torch.no_grad():
        z = prefix_logits(student, row, position)
        fields, profiles = {}, {}
        for target, values in requests.items():
            value = values[index]
            q = value["q_logits"].to(z.device)
            profiles[target], live = profile_and_signal(z, q)
            fields[f"{target}/live"] = live.cpu()
            fields[f"{target}/frozen_B0"] = value["B0"]
    result = measure_site(student, row, position, fields, iterations=iterations, seed=seed)
    spectrum = result.pop("spectrum")
    record = dict(
        row_id=row["row_id"],
        position=position,
        split=row.get("split"),
        domain=row["domain"],
        group=row.get("group"),
        profiles=profiles,
        **result,
        spectrum={
            key: value.tolist() if isinstance(value, torch.Tensor) else value
            for key, value in spectrum.items()
            if key != "vectors"
        },
    )
    return dict(record=record, spectrum=spectrum, requests=fields)


def measure_cross(student, source_direction, receiver_row, position, request, *, gram=None):
    """One source stays on device; receiver graphs have pair-local lifetimes."""
    operator = LogitJacobian(prefix_logits(student, receiver_row, position), student.parameters())
    response = operator.pushforward(source_direction)
    metrics = response_summary(request.to(response.device), response)
    if gram is None:
        receiver_direction = operator.pullback(request.to(response.device))
        gram = direction_inner(receiver_direction, source_direction)
    return dict(
        **metrics,
        gradient_inner=gram,
        duality_relative_error=abs(metrics["request_response_inner"] - gram) / max(abs(gram), 1e-20),
    )


def measure_cross_panel(student, selected, indices, payloads, store, *, stage):
    """Source-first streaming: O(parameter count) transient direction storage.

    Checkpoint each scalar pair. Resume regenerates at most one source VJP,
    not spectra or completed pairs. Reverse pairs reuse the symmetric Gram.
    """
    cross = []
    gram_cache = {}
    for source in range(len(indices)):
        row_s, pos_s = selected[indices[source]]
        for field_index, (field, request_s) in enumerate(payloads[source]["requests"].items()):
            key = f"cross-{source:03d}-{field_index:03d}"
            completed = store.json(key) or []
            by_receiver = {item["receiver"]: item for item in completed}
            missing = [r for r in range(len(indices)) if r != source and r not in by_receiver]
            if missing:
                started = time.monotonic()
                z = prefix_logits(student, row_s, pos_s)
                direction = logit_pullback(z, student.parameters(), request_s.to(z.device))
                del z  # Source forward graph is freed before receiver graphs.
                progress(
                    "logit_source_ready",
                    stage=stage,
                    source=source,
                    field=field,
                    elapsed=time.monotonic() - started,
                    direction_bytes=sum(g.numel() * g.element_size() for g in direction if g is not None),
                )
                for receiver in missing:
                    started = time.monotonic()
                    row_r, pos_r = selected[indices[receiver]]
                    pair_key = (min(source, receiver), max(source, receiver), field)
                    metrics = measure_cross(
                        student,
                        direction,
                        row_r,
                        pos_r,
                        payloads[receiver]["requests"][field],
                        gram=gram_cache.get(pair_key),
                    )
                    item = dict(
                        source=source,
                        receiver=receiver,
                        field=field,
                        **metrics,
                        source_row_id=row_s["row_id"],
                        receiver_row_id=row_r["row_id"],
                        source_split=row_s.get("split"),
                        receiver_split=row_r.get("split"),
                        elapsed_seconds=time.monotonic() - started,
                    )
                    by_receiver[receiver] = item
                    gram_cache[pair_key] = metrics["gradient_inner"]
                    store.save_json(key, [by_receiver[r] for r in sorted(by_receiver)])
                    progress(
                        "logit_pair_done",
                        stage=stage,
                        source=source,
                        receiver=receiver,
                        field=field,
                        elapsed=item["elapsed_seconds"],
                    )
                del direction
            for receiver in sorted(by_receiver):
                item = by_receiver[receiver]
                gram_cache[(min(source, receiver), max(source, receiver), field)] = item["gradient_inner"]
                cross.append(item)
    return cross


def run_task(task, selected, requests, indices, *, destination, signature, iterations, seed):
    # A signature binds both execution precision and input bank. A lock keeps
    # two launchers from calculating/writing the same expensive work unit.
    with CheckpointStore(destination.with_suffix(".parts"), signature) as store:
        if destination.exists():
            if json.loads(destination.read_text())["signature"] != json.loads(json.dumps(signature)):
                raise ValueError(f"Stale logit result: {destination}")
            return
        student = load_model(task["student"], checkpointing=False)
        if next(student.parameters()).is_cuda:
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        try:
            # Fused SDPA does not support the reverse-over-reverse JVP.
            with sdpa_kernel(SDPBackend.MATH):
                payloads = []
                for site, index in enumerate(indices):
                    payload = store.tensors(f"site-{site:03d}")
                    if payload is None:
                        row, pos = selected[index]
                        payload = site_payload(
                            student, row, pos, requests, index, iterations=iterations, seed=seed + site
                        )
                        store.save_tensors(f"site-{site:03d}", payload)
                    payloads.append(payload)
                    progress(
                        "logit_site_done", stage=task["stage"], site=site, elapsed=payload["record"]["elapsed_seconds"]
                    )
                cross = measure_cross_panel(student, selected, indices, payloads, store, stage=task["stage"])
            output = dict(
                signature=signature,
                task=task,
                sites=[p["record"] for p in payloads],
                cross=cross,
                precision="FP32-master/BF16-autocast/math-SDPA",
                execution=dict(
                    direction_storage="one-source-field-on-device",
                    elapsed_seconds=time.monotonic() - started,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated() if next(student.parameters()).is_cuda else 0,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved() if next(student.parameters()).is_cuda else 0,
                ),
            )
            artifact = dict(
                signature=signature,
                spectra=[p["spectrum"] for p in payloads],
                requests=[p["requests"] for p in payloads],
            )
            write_tensors(destination.with_suffix(".pt"), artifact)
            write_json(destination, output)
        finally:
            del student
            collect_memory()


def select_indices(selected, requests, *, positions, max_sites):
    """Balance questions/splits before adding more positions from a question."""
    candidates = {}
    for i, (row, pos) in enumerate(selected):
        if row["domain"] == "math":
            candidates.setdefault(row["row_id"], []).append(i)
    by_split = {}
    for members in candidates.values():
        if positions == "pressure":
            chosen = {
                max(members, key=lambda i: float(values[i]["B0"].double().square().sum()))
                for values in requests.values()
            }
            members = sorted(chosen)
        elif positions == "middle":
            members = [i for i in members if selected[i][1] == len(selected[i][0]["response_ids"]) // 2]
        split = selected[members[0]][0].get("split") if members else None
        if members:
            by_split.setdefault(split, []).append(members)
    if not by_split:
        raise ValueError("No Math prefixes selected")
    questions = [
        items[j] for j in range(max(map(len, by_split.values()))) for items in by_split.values() if j < len(items)
    ]
    return [items[j] for j in range(max(map(len, questions))) for items in questions if j < len(items)][:max_sites]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default="r1")
    parser.add_argument("--branch", required=True)
    parser.add_argument("--bank")
    parser.add_argument("--stages", type=int, nargs="+", default=[0, 80, 240])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--request-cache", type=Path, help="Reuse an existing prefix requests.pt directory")
    parser.add_argument("--max-sites", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--positions", choices=["middle", "all", "pressure"], default="middle")
    args = parser.parse_args()
    if args.max_sites < 1:
        parser.error("--max-sites must be positive")
    torch.set_num_threads(4)
    manifest = json.loads(args.manifest.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    tasks, rows, selected, requests, identity = prepare_panel(
        manifest,
        args.manifest.parent,
        model=args.model,
        bank=args.bank,
        out=args.request_cache or args.out / "requests",
        seed=args.seed,
    )
    indices = select_indices(selected, requests, positions=args.positions, max_sites=args.max_sites)
    for task in branch_tasks(tasks, args.branch):
        if task["stage"] not in args.stages:
            continue
        destination = args.out / f"{args.branch}-s{task['stage']:03d}.json"
        signature = dict(
            protocol="logit-kernel-v2",
            request=identity,
            checkpoint=checkpoint_identity(task["student"]),
            indices=indices,
            iterations=args.iterations,
            seed=args.seed,
            positions=args.positions,
            attention="math",
            sign="positive-loss-gradient",
        )
        if destination.exists():
            if json.loads(destination.read_text())["signature"] != json.loads(json.dumps(signature)):
                raise ValueError(f"Stale logit result: {destination}")
            continue
        run_task(
            task,
            selected,
            requests,
            indices,
            destination=destination,
            signature=signature,
            iterations=args.iterations,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
