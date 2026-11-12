# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Token-local amplitude controls using the production RKL logit derivatives."""

import json

import torch

from verl.trainer.distillation.rows import precompute_kl_rows

from .checkpoints import CheckpointStore
from .inputs import source_fingerprint
from .reductions import response_denominator
from .runtime import checkpoint_identity, teacher_cache, write_json
from .trajectory import panel, parser, tasks


def alignment_moments(live, initial):
    """Separate directional rotation from heterogeneous per-token rescaling.

    P0-zero positions use alpha=0: any live request there is novel energy.
    A negative local scale counts as direction reversal, not amplitude fitting.
    """
    p, p0 = live.double().square().sum(-1), initial.double().square().sum(-1)
    dot = (live.double() * initial.double()).sum(-1)
    projected = torch.where(p0 > 0, dot.square() / p0.clamp_min(1e-300), 0.0)
    perpendicular = (p - projected).clamp_min(0)
    positive_residual = torch.where(dot > 0, perpendicular, p)
    return torch.stack((p, p0, dot, (p * p0).sqrt(), perpendicular, positive_residual), -1)


def run(task, initial_path, root, out, *, token_chunk):
    rows, math, _ = panel(task)
    signature = dict(
        protocol="token-amplitude-control-v1",
        task=task,
        student=checkpoint_identity(task["student"]),
        initial=checkpoint_identity(initial_path),
        teacher=checkpoint_identity(task["teacher"]),
        records=source_fingerprint(math),
        token_chunk=token_chunk,
    )
    destination = out / f"{task['target']}-s{task['stage']:03d}.alignment.json"
    with CheckpointStore(destination.with_suffix(".parts"), signature) as store:
        if destination.exists():
            if json.loads(destination.read_text())["signature"] != signature:
                raise ValueError(f"Stale result {destination}")
            return
        current = teacher_cache(root / "cache", task["student"], math, student=True)
        initial = teacher_cache(root / "cache", initial_path, math, student=True)
        teacher = teacher_cache(root / "cache", task["teacher"], rows)
        w, w0, wq = [payload["head"].cuda() for payload in (current, initial, teacher)]
        records = []
        with torch.no_grad():
            for row in math:
                key = row["row_id"]
                value = store.json("row-" + key)
                if value is None:
                    h, h0, hq = [payload["hidden"][key].cuda() for payload in (current, initial, teacher)]
                    moment = torch.zeros(6, dtype=torch.float64, device="cuda")
                    for start in range(0, len(h), token_chunk):
                        sl = slice(start, start + token_chunk)
                        _, b = precompute_kl_rows(h[sl], w, hq[sl], wq)
                        _, b0 = precompute_kl_rows(h0[sl], w0, hq[sl], wq)
                        moment += alignment_moments(b, b0).sum(0)
                    value = dict(
                        row_id=key,
                        split=row["split"],
                        group=row["group"],
                        token_count=len(h),
                        denominator=response_denominator(len(h), task["cap"], task.get("reduction", "fixed_cap")),
                        moments=(moment / response_denominator(len(h), task["cap"], task.get("reduction", "fixed_cap")))
                        .cpu()
                        .tolist(),
                    )
                    store.save_json("row-" + key, value)
                records.append(value)
        write_json(destination, dict(signature=signature, records=records))
        print(json.dumps(dict(event="alignment_finished", task=task["id"])), flush=True)


def main():
    args = parser(__doc__).parse_args()
    if args.max_groups is not None:
        raise ValueError("alignment uses the complete panel")
    initial, selected = tasks(args)
    for task in selected:
        run(task, initial, args.manifest.parent, args.out, token_chunk=args.token_chunk)


if __name__ == "__main__":
    main()
