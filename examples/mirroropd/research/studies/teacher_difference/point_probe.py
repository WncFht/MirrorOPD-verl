# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact sampled-prefix PAC with live and frozen initial request fields.

This probe measures the paper's prefix-level A/C on a declared finite prefix
population. It does not infer a full Fisher or local kernel eigensystem.
"""

import argparse
import json
from pathlib import Path

import torch

from .inputs import source_fingerprint
from .metrics import moment_summary, tensor_norm_sq
from .prefixes import branch_tasks, prepare_panel, select  # noqa: F401 -- compatibility exports
from .runtime import (
    checkpoint_identity,
    collect_memory,
    gradient_norms,
    load_model,
    prefix_logits,
    write_json,
)
from .signals import fisher_diagnostics, position_signal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bank")
    parser.add_argument("--branch", required=True)
    parser.add_argument("--stages", type=int, nargs="+")
    parser.add_argument("--trajectories-per-stratum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.parent
    suffix = f"-{args.bank}" if args.bank else ""
    out = root / "point-probe" / f"{args.model}-{args.branch}{suffix}"
    tasks, rows, selected, requests, request_identity = prepare_panel(
        manifest,
        root,
        model=args.model,
        bank=args.bank,
        out=out,
        per_stratum=args.trajectories_per_stratum,
        seed=args.seed,
    )
    branch = [t for t in branch_tasks(tasks, args.branch) if args.stages is None or t["stage"] in args.stages]
    for task in branch:
        destination = out / f"s{task['stage']:03d}.json"
        signature = dict(request=request_identity, checkpoint=checkpoint_identity(task["student"]))
        if destination.exists():
            if json.loads(destination.read_text()).get("signature") != json.loads(json.dumps(signature)):
                raise ValueError(f"Stale prefix result: {destination}")
            continue
        teacher_copy = load_model(task["student"], trainable=False)
        with torch.no_grad():
            copy_logits = [prefix_logits(teacher_copy, row, pos).cpu() for row, pos in selected]
        del teacher_copy
        collect_memory()
        student = load_model(task["student"], checkpointing=task.get("checkpointing", False))
        precision_reference = []
        with torch.no_grad():
            for (row, pos), qcopy in zip(selected, copy_logits, strict=True):
                z = prefix_logits(student, row, pos)
                kl, b = position_signal(z, qcopy.to("cuda"))
                precision_reference.append(dict(rkl=float(kl), pressure=float(b.double().square().sum())))
        output = dict(
            signature=signature,
            student=task["student"],
            stage=task["stage"],
            branch=args.branch,
            model=args.model,
            source_fingerprint=source_fingerprint(rows),
            prefixes=[
                dict(row_id=r["row_id"], domain=r["domain"], split=r.get("split"), position=p) for r, p in selected
            ],
            fields={},
            teacher_copy_reference=precision_reference,
            copy_reference_note=(
                "Same checkpoint cast to BF16 teacher versus FP32-master/BF16-autocast student; "
                "numerical/quantization control, not a universal risk lower bound."
            ),
        )
        for target, values in requests.items():
            for field in ["live", "frozen_B0"]:
                student.zero_grad(set_to_none=True)
                pressure = []
                second = []
                losses = []
                fisher = []
                with gradient_norms(student) as buffer:
                    for (row, pos), value in zip(selected, values, strict=True):
                        buffer.zero_()
                        z = prefix_logits(student, row, pos)
                        with torch.no_grad():
                            kl, live = position_signal(z.detach(), value["q_logits"].to("cuda"))
                            B = live if field == "live" else value["B0"].to("cuda")
                            pressure.append(float(B.double().square().sum()))
                            losses.append(float(kl))
                            diagnostics = fisher_diagnostics(z.detach()[None], value["q_logits"].to("cuda")[None])
                            fisher.append({k: float(v[0]) for k, v in diagnostics.items()})
                        (z * B.detach()).sum().backward()
                        second.append(float(buffer))
                total = sum(float(tensor_norm_sq(p.grad)) for p in student.parameters() if p.grad is not None)
                summary = moment_summary(
                    pressure=sum(pressure) / len(pressure),
                    second=sum(second) / len(second),
                    mean_norm=total / len(selected) ** 2,
                    samples=len(selected),
                )
                output["fields"][f"{target}/{field}"] = dict(
                    summary=summary, pressure=pressure, gradient_norm_sq=second, live_kl=losses, live_fisher=fisher
                )
                print(
                    json.dumps(
                        dict(
                            event="point_field_done",
                            stage=task["stage"],
                            branch=args.branch,
                            target=target,
                            field=field,
                            summary=summary,
                        )
                    ),
                    flush=True,
                )
        write_json(destination, output)
        del student
        collect_memory()


if __name__ == "__main__":
    main()
