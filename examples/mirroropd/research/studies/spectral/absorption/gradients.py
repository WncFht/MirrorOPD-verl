# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact per-site RKL gradients and direction/coherence readouts."""

import time

import torch

from examples.mirroropd.research.backends.direction_projection import DirectionProjector
from examples.mirroropd.research.core.io import dump

from ...teacher_difference.runtime import checkpoint_identity, collect_memory, load_model, prefix_logits
from .journal import GroupJournal
from .performance import PerformanceLedger
from .protocol import save_tensor


def gradients(context, job, out, banks, reference, teacher_head, records):
    # Evaluate exact per-example RKL gradients on all attention/MLP coordinates.
    # These are diagnostic generated states, not historical training rollouts.
    projector = DirectionProjector(out / "directions", backend=context.projection)
    for stage, path in job["gradient_stages"].items():
        # A separate scope drops parameter dictionaries, leaf gradients and the
        # last loss graph before allocating the next model's FP32 weights.
        _gradient_stage(context, job, out, banks, reference, teacher_head, records, projector, stage, path)
        collect_memory()


def _gradient_stage(context, job, out, banks, reference, teacher_head, records, projector, stage, path):
    destination = out / "gradients" / stage
    if (destination / "complete.json").exists():
        return
    performance = PerformanceLedger(destination)
    journal = GroupJournal(
        destination,
        dict(
            job=job,
            stage=stage,
            model=checkpoint_identity(path),
            banks=banks,
            records=records,
            projection=context.projection,
        ),
    )
    with performance.phase("model_load", synchronize=torch.cuda.synchronize):
        model = load_model(path, trainable=True, checkpointing=True, device="cuda")
    model.requires_grad_(False)
    params = dict(model.named_parameters())
    for record in records:
        params[record["key"]].requires_grad_(True)
    totals = {}
    norm_sums = {}
    squared_sums = {}
    projection_rows = [r for group in journal.completed.values() for r in group["projections"]]
    teacher_head = teacher_head.cuda()
    site_count = len({(r["origin"], r["row_id"], r["position"]) for r in projection_rows})
    resumed_sites = site_count
    current_group = None
    all_summary = [r for group in journal.completed.values() for r in group["summaries"]]
    timings = []

    def flush_group(
        current_group,
        destination=destination,
        totals=totals,
        norm_sums=norm_sums,
        squared_sums=squared_sums,
        all_summary=all_summary,
        projection_rows=projection_rows,
        journal=journal,
        performance=performance,
    ):
        if not totals:
            return
        summary = []
        with performance.phase("group_norms"):
            for (group, key), total in totals.items():
                norm = float(total.double().norm())
                summary.append(
                    dict(
                        group=group,
                        key=key,
                        sum_gradient_norm=norm,
                        sum_individual_norm=norm_sums[group, key],
                        sum_individual_squared_norm=squared_sums[group, key],
                        vector_coherence=norm / norm_sums[group, key] if norm_sums[group, key] else None,
                    )
                )
        with performance.phase("group_tensor_write"):
            save_tensor(destination / "domain-gradient-sums" / (current_group.replace(":", "_") + ".pt"), totals)
        group_rows = [r for r in projection_rows if f"{r['origin']}:{r['split']}:{r['domain']}" == current_group]
        with performance.phase("group_commit"):
            journal.save(current_group, group_rows, summary)
        all_summary.extend(summary)
        totals.clear()
        norm_sums.clear()
        squared_sums.clear()

    for origin, rows in banks.items():
        selected = []
        for split in ["calibration", "heldout"]:
            for domain in sorted({r["domain"] for r in rows}):
                selected += [r for r in rows if r["split"] == split and r["domain"] == domain][:2]
        for row in selected:
            for position in sorted({0, len(row["response_ids"]) - 1}):
                context.check_stop()
                group = f"{origin}:{row['split']}:{row['domain']}"
                if group in journal.completed:
                    continue
                if group != current_group:
                    flush_group(current_group)
                    current_group = group
                started = time.perf_counter()
                with performance.phase("forward_backward", synchronize=torch.cuda.synchronize):
                    model.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        lt = torch.nn.functional.linear(
                            reference[origin, row["row_id"]][position].cuda(), teacher_head
                        ).log_softmax(-1)
                    ls = prefix_logits(model, row, position).log_softmax(-1)
                    loss = (ls.exp() * (ls - lt)).sum()
                    loss.backward()
                    loss_value = float(loss.detach())
                backward_done = time.perf_counter()
                group = f"{origin}:{row['split']}:{row['domain']}"
                for record in records:
                    context.check_stop()
                    key = record["key"]
                    raw_gradient = params[key].grad.detach()
                    with performance.phase("direction_read_transfer_projection"):
                        norm, values = projector.project(record["index"], raw_gradient)
                    with performance.phase("gradient_d2h"):
                        grad = raw_gradient.float().cpu()
                    with performance.phase("cpu_accumulate"):
                        identifier = (group, key)
                        if identifier not in totals:
                            totals[identifier] = torch.zeros_like(grad)
                            norm_sums[identifier] = 0.0
                            squared_sums[identifier] = 0.0
                        totals[identifier].add_(grad)
                        norm_sums[identifier] += norm
                        squared_sums[identifier] += norm**2
                    projection_rows.append(
                        dict(
                            stage=stage,
                            origin=origin,
                            split=row["split"],
                            domain=row["domain"],
                            row_id=row["row_id"],
                            position=position,
                            key=key,
                            gradient_norm=norm,
                            reverse_kl=loss_value,
                            directions=values,
                        )
                    )
                timings.append(
                    dict(
                        origin=origin,
                        row_id=row["row_id"],
                        position=position,
                        backward_seconds=backward_done - started,
                        projection_transfer_seconds=time.perf_counter() - backward_done,
                    )
                )
                dump(destination / "timings.json", timings)
                site_count += 1
                dump(destination / "progress.json", dict(sites=site_count, rows=len(projection_rows)))
                performance.snapshot(sites=site_count - resumed_sites)
    flush_group(current_group)
    dump(destination / "per-site-projections.json", projection_rows)
    dump(destination / "vector-coherence.json", all_summary)
    dump(
        destination / "complete.json",
        dict(
            projection_backend=context.projection,
            sites=site_count,
            stage=stage,
            state_source="newly generated frozen prefixes; historical train rollout not substituted",
            optimizer_state="not available in this job; no Adam claim",
        ),
    )
    performance.snapshot(sites=site_count - resumed_sites, complete=True)
    model.zero_grad(set_to_none=True)
    totals.clear()
