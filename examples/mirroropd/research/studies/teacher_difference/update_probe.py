# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Equal-displacement component updates evaluated on the native RKL objective.

Source groups and displacement grid are fixed before test evaluation. The first
six heldout B8 groups calibrate the displacement independently per direction;
the remaining six groups assess the chosen step. Every trial restores the same
FP32 master parameters exactly, avoiding accumulated roundoff across trials.
"""

import json
import time

import torch

from verl.trainer.distillation.rows import precompute_kl_rows

from .checkpoints import CheckpointStore
from .inputs import source_fingerprint, validate_source_exposure
from .metrics import squared_norm
from .reductions import response_denominator
from .response import GradientBank, cotangents
from .runtime import checkpoint_identity, flatten_tensors, hidden_states, load_model, write_json
from .trajectory import panel, parser, request_cache, select_source_groups, tasks


@torch.no_grad()
def restore_step(model, baseline, direction, scale):
    """Restore parameters from an immutable CPU master and add one direction."""
    offset = 0
    for parameter in model.parameters():
        size = parameter.numel()
        parameter.copy_(baseline[offset : offset + size].view_as(parameter))
        if scale:
            parameter.add_(direction[offset : offset + size].view_as(parameter), alpha=-scale)
        offset += size


@torch.no_grad()
def evaluate(model, teacher, rows, *, cap, token_chunk, reduction="fixed_cap"):
    values = []
    for row in rows:
        key = row["row_id"]
        hs = hidden_states(model, row["prompt_ids"] + row["response_ids"][:-1])[len(row["prompt_ids"]) - 1 :][:cap]
        ht = teacher["hidden"][key][: len(hs)].to(hs.device)
        loss = torch.zeros((), dtype=torch.float64, device=hs.device)
        for start in range(0, len(hs), token_chunk):
            sl = slice(start, start + token_chunk)
            costs, _ = precompute_kl_rows(hs[sl], model.lm_head.weight, ht[sl], teacher["head"])
            loss.add_(costs.double().sum())
        values.append(
            dict(
                row_id=key,
                group=row.get("diagnostic_group", row.get("group")),
                split=row["split"],
                loss=float(loss / response_denominator(len(hs), cap, reduction)),
            )
        )
    return values


def source_directions(
    model, teacher, initial, rows, *, cap, token_chunk, fields=("live", "parallel"), reduction="fixed_cap"
):
    bank = GradientBank(model, fields)
    for row in rows:
        key = row["row_id"]
        model.zero_grad(set_to_none=True)
        hs = hidden_states(model, row["prompt_ids"] + row["response_ids"][:-1])[len(row["prompt_ids"]) - 1 :][:cap]
        ht, h0 = [payload["hidden"][key][: len(hs)].to(hs.device) for payload in (teacher, initial)]
        for index, (field, dh, dw, _) in enumerate(
            cotangents(
                hs,
                model.lm_head.weight,
                ht,
                teacher["head"],
                (h0, initial["head"]),
                fields=list(fields),
                cap=cap,
                reduction=reduction,
                token_chunk=token_chunk,
                bundle=min(3, len(fields)),
            )
        ):
            model.zero_grad(set_to_none=True)
            torch.autograd.backward((hs, model.lm_head.weight), (dh, dw), retain_graph=index < len(fields) - 1)
            bank.add(field)
    model.zero_grad(set_to_none=True)
    result = {field: bank.export(field, len(rows)) for field in fields}
    del bank
    result["native"] = result.pop("live")
    if "perpendicular" not in result and "parallel" in result:
        result["perpendicular"] = result["native"] - result["parallel"]
    return result


def run(task, initial_path, root, out, *, source_indices, displacements, token_chunk):
    rows, math, groups = panel(task)
    held = [(label, batch) for label, batch in groups if label[0] == "heldout"]
    selected = select_source_groups(groups, source_indices)
    validate_source_exposure(task, [r for _, batch in selected for r in batch])
    cal = {label[1] for label, _ in held[: len(held) // 2]}
    signature = dict(
        protocol="component-equal-displacement-v1",
        task=task,
        student=checkpoint_identity(task["student"]),
        teacher=checkpoint_identity(task["teacher"]),
        initial=checkpoint_identity(initial_path),
        records=source_fingerprint(math),
        source_indices=source_indices,
        displacements=displacements,
        calibration=sorted(cal),
        token_chunk=token_chunk,
        directions=["native", "parallel", "perpendicular"],
    )
    destination = out / f"{task['target']}-s{task['stage']:03d}.updates.json"
    with CheckpointStore(destination.with_suffix(".parts"), signature) as store:
        if destination.exists():
            if json.loads(destination.read_text())["signature"] != signature:
                raise ValueError(f"Stale result {destination}")
            return
        started = time.monotonic()
        teacher, initial = request_cache(task, initial_path, root, rows, math)
        model = load_model(task["student"])
        parameters = tuple(model.parameters())
        baseline = flatten_tensors(parameters, [p.detach() for p in parameters])
        baseline_loss = store.json("baseline")
        if baseline_loss is None:
            baseline_loss = evaluate(
                model,
                teacher,
                math,
                cap=task["cap"],
                reduction=task.get("reduction", "fixed_cap"),
                token_chunk=token_chunk,
            )
            store.save_json("baseline", baseline_loss)
        results = []
        for label, batch in selected:
            source_key = label[1]
            directions = source_directions(
                model,
                teacher,
                initial,
                batch,
                cap=task["cap"],
                reduction=task.get("reduction", "fixed_cap"),
                token_chunk=token_chunk,
            )
            evaluation = [row for _, batch_h in held for row in batch_h] + batch
            for field, direction in directions.items():
                norm = squared_norm(direction) ** 0.5
                if norm <= 0:
                    raise ValueError(f"Zero candidate direction {field}")
                device_direction = direction.cuda()
                for delta in displacements:
                    key = f"{source_key}-{field}-{delta}"
                    value = store.json(key)
                    if value is None:
                        restore_step(model, baseline, device_direction, delta / norm)
                        values = evaluate(
                            model,
                            teacher,
                            evaluation,
                            cap=task["cap"],
                            reduction=task.get("reduction", "fixed_cap"),
                            token_chunk=token_chunk,
                        )
                        restore_step(model, baseline, device_direction, 0)
                        value = dict(
                            source=source_key, field=field, displacement=delta, gradient_norm=norm, records=values
                        )
                        store.save_json(key, value)
                    results.append(value)
                    print(
                        json.dumps(
                            dict(
                                event="update_trial",
                                branch=task["target"],
                                source=source_key,
                                field=field,
                                displacement=delta,
                                seconds=time.monotonic() - started,
                            )
                        ),
                        flush=True,
                    )
                del device_direction
            del directions
        restore_step(model, baseline, None, 0)
        write_json(
            destination,
            dict(signature=signature, baseline=baseline_loss, trials=results, seconds=time.monotonic() - started),
        )


def main():
    arguments = parser(__doc__)
    arguments.add_argument("--source-indices", type=int, nargs="+", default=[15, 17, 19])
    arguments.add_argument("--displacements", type=float, nargs="+", default=[0.003, 0.01, 0.03])
    args = arguments.parse_args()
    if args.max_groups is not None or min(args.displacements) <= 0:
        arguments.error("complete panel and positive displacements required")
    initial, selected = tasks(args)
    for task in selected:
        run(
            task,
            initial,
            args.manifest.parent,
            args.out,
            source_indices=args.source_indices,
            displacements=args.displacements,
            token_chunk=args.token_chunk,
        )


if __name__ == "__main__":
    main()
