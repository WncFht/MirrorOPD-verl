# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""H4096 fixed held-out diagnostics with exact acceptance of the original H1024 token prefix."""

import argparse
import gc
from dataclasses import replace
from pathlib import Path

from examples.mirroropd.research.backends import hf_models, hf_rkl
from examples.mirroropd.research.core.artifacts import complete, file_hash, receipt
from examples.mirroropd.research.core.io import dump, read, write_csv
from examples.mirroropd.research.studies.fixed_transfer import data
from examples.mirroropd.research.studies.fixed_transfer.spec import Execution, Study, fingerprint
from examples.mirroropd.research.studies.fixed_transfer.training import (
    arm_by_name,
    checkpoint_directory,
    validate_checkpoint,
)

GROUPS = ("M01", "M02", "M03", "M04", "Code01", "Code02", "IF01", "IF02")
MODULE = "examples.mirroropd.research.studies.fixed_transfer.long_panel"


def extended(study):
    if study.training.horizon != 1024:
        raise ValueError("This panel extends an established H1024 study")
    return replace(
        study,
        name=study.name + "-h4096-heldout",
        output=str(study.directory) + "-h4096-heldout",
        arms=(),
        continuation=None,
        training=replace(study.training, horizon=4096),
    )


def selected_rows(study):
    prompts = read(study.directory / "prepared/bank-prompts.json")
    root = data.bank_directory(study)
    rows = []
    for index in range(len(prompts) // study.training.batch_size):
        batch = prompts[index * study.training.batch_size : (index + 1) * study.training.batch_size]
        if batch[0]["group"] in GROUPS:
            rows.extend(read(root / f"batch-{index:03d}.json"))
    if len(rows) != 128 or len({r["id"] for r in rows}) != 128:
        raise ValueError("Long panel requires exactly the fixed 128 held-out prompts")
    expected = {r["id"]: r for r in prompts if r["group"] in GROUPS}
    bank_key = read(study.directory / "prepared/complete.json")["bank_key"]
    if {r["id"] for r in rows} != set(expected):
        raise ValueError("Long panel prompt identities differ")
    for row in rows:
        if row["source"] != bank_key or row["cap"] != 4096:
            raise ValueError("Wrong long-bank identity or response budget")
        if any(row[k] != expected[row["id"]][k] for k in ["prompt_ids", "group", "domain"]):
            raise ValueError("Long-bank prompt content changed")
        tokens = row["response_ids"]
        if not 0 < len(tokens) <= 4096 or (len(tokens) < 4096 and tokens[-1] not in study.evaluation.stop_ids):
            raise ValueError("Long response is incomplete without EOS")
    return rows


def verify_prefixes(short, long):
    reference = {r["id"]: r for r in short}
    failures = []
    for row in long:
        old = reference[row["id"]]
        if (
            row["prompt_ids"] != old["prompt_ids"]
            or row["response_ids"][: len(old["response_ids"])] != old["response_ids"]
        ):
            failures.append(row["id"])
    return dict(prompts=len(long), matched=len(long) - len(failures), mismatched_ids=failures)


def score_inputs(study, arm, long):
    from examples.mirroropd.research.studies.fixed_transfer.runner import check_inputs
    from examples.mirroropd.research.studies.fixed_transfer.training import scientific_inputs

    checkpoint = checkpoint_directory(study, arm, arm.updates)
    check_inputs(checkpoint / "complete.json", dict(scientific_inputs(study, arm), step=arm.updates))
    saved = validate_checkpoint(checkpoint)
    return dict(
        student=study.models[study.student].identity,
        teacher=study.models[arm.teacher].identity,
        weights=saved["weight_identity"],
        step=arm.updates,
        bank=read(long.directory / "prepared/complete.json")["bank_key"],
        groups=GROUPS,
        horizon=4096,
        source=file_hash(__file__),
        engine=file_hash(hf_rkl.__file__),
        backend=file_hash(hf_models.__file__),
    )


def jobs(study):
    long = extended(study)
    data.prepare(long)
    result = []
    for shard in range(4):
        directory, inputs = data.bank_shard_receipt(long, shard, groups=GROUPS)
        result.append(
            dict(
                name=f"long-bank-{shard}",
                module=MODULE,
                args=["bank", "--shard", str(shard)],
                dependencies=[],
                retries=0,
                marker=str(directory / "complete.json"),
                fingerprint=fingerprint(inputs),
            )
        )
    for arm in study.arms:
        result.append(
            dict(
                name="long-score-" + arm.name,
                module=MODULE,
                args=["score", "--arm", arm.name],
                dependencies=[f"long-bank-{shard}" for shard in range(4)],
                retries=0,
                marker=str(long.directory / "arms" / arm.name / "complete.json"),
                fingerprint=fingerprint(score_inputs(study, arm, long)),
            )
        )
    return result


def score(study, name):
    import torch

    arm = arm_by_name(study, name)
    long = extended(study)
    inputs = score_inputs(study, arm, long)
    destination = long.directory / "arms" / name
    if complete(destination, inputs):
        return
    rows = selected_rows(long)
    acceptance = verify_prefixes(data.bank_rows(study), rows)
    dump(destination / "prefix-acceptance.json", acceptance)
    if acceptance["mismatched_ids"]:
        raise ValueError("Long rollout does not preserve the established S0 prefix")
    weights = checkpoint_directory(study, arm, arm.updates)
    validate_checkpoint(weights, verify_hash=True)
    teacher = hf_models.load(study.models[arm.teacher])
    head = teacher.get_output_embeddings().weight.detach().float().clone()
    with torch.inference_mode():
        states = {r["id"]: hf_models.hidden(teacher, [r], pad_id=study.evaluation.stop_ids[0]).cpu() for r in rows}
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    snapshots = []
    for step, path in [(0, None), (arm.updates, weights)]:
        model = hf_models.load(study.models[study.student], trainable=True, path=path)
        measured = hf_rkl.score(
            model, head, states, rows, step, long.training, Execution(microbatch=1), study.evaluation.stop_ids[0]
        )
        dump(destination / f"fixed-{step:03d}.json", measured)
        snapshots.append(measured)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    summary = []
    for band, width in [("early", 1024), ("middle", 1024), ("late", 2048), ("all", 4096)]:
        for metric in ["rkl", "js"]:
            initial, final = [sum(r["stats"][band][metric] for r in snapshot) / (128 * width) for snapshot in snapshots]
            summary.append(
                dict(
                    band=band,
                    metric=metric,
                    budget_per_prompt=width,
                    prompts=128,
                    actual_tokens=sum(r["stats"][band]["tokens"] for r in snapshots[0]),
                    initial=initial,
                    final=final,
                    drop_percent=100 * (1 - final / initial) if initial > 0 else None,
                )
            )
    write_csv(destination / "risks.csv", summary)
    receipt(
        destination,
        inputs,
        prefix_acceptance=acceptance,
        scientific_updates=0,
        interpretation="frozen S0 held-out extension; early=0:1024, middle=1024:2048, late=2048:4096",
    )


def audit(study, name):
    """CPU audit of coverage and short-prefix risk differences; equality is measured, not assumed."""
    import csv
    import math

    arm = arm_by_name(study, name)
    long = extended(study)
    destination = long.directory / "arms" / name
    completed = read(destination / "complete.json")
    rows = selected_rows(long)
    prefix = verify_prefixes(data.bank_rows(study), rows)
    assert prefix == completed["prefix_acceptance"] and not prefix["mismatched_ids"]
    assert completed["scientific_updates"] == 0 and completed["inputs"]["step"] == arm.updates
    result = dict(
        prefix=prefix, step=arm.updates, panels={}, short_prefix={}, scoring_claim="measured numerical differences"
    )
    snapshots = {}
    for step in [0, arm.updates]:
        snapshots[step] = read(destination / f"fixed-{step:03d}.json")
        assert [r["id"] for r in snapshots[step]] == [r["id"] for r in rows]
        old = {r["id"]: r for r in read(study.directory / "arms" / name / f"fixed-{step:03d}.json")}
        for measured, bank in zip(snapshots[step], rows, strict=True):
            assert measured["source"] == bank["source"] and measured["length"] == len(bank["response_ids"])
            for band, lo, hi in [("early", 0, 1024), ("middle", 1024, 2048), ("late", 2048, 4096), ("all", 0, 4096)]:
                assert measured["stats"][band]["tokens"] == max(0, min(len(bank["response_ids"]), hi) - lo)
        early = [r["stats"]["early"]["rkl"] / 1024 for r in snapshots[step]]
        short = [old[r["id"]]["stats"]["all"]["rkl"] / 1024 for r in snapshots[step]]
        error = [abs(a - b) for a, b in zip(early, short, strict=True)]
        result["short_prefix"][str(step)] = dict(
            short_source=str(study.directory / "arms" / name / f"fixed-{step:03d}.json"),
            short_sha256=file_hash(study.directory / "arms" / name / f"fixed-{step:03d}.json"),
            long_source=str(destination / f"fixed-{step:03d}.json"),
            long_sha256=file_hash(destination / f"fixed-{step:03d}.json"),
            old_mean=sum(short) / len(short),
            long_early_mean=sum(early) / len(early),
            max_absolute_error=max(error),
            mean_absolute_error=sum(error) / len(error),
            aggregate_relative_change_percent=100 * (sum(early) / sum(short) - 1),
        )
    with (destination / "risks.csv").open() as stream:
        for row in csv.DictReader(stream):
            band, metric = row["band"], row["metric"]
            for step, key in [(0, "initial"), (arm.updates, "final")]:
                value = sum(r["stats"][band][metric] for r in snapshots[step]) / (128 * int(row["budget_per_prompt"]))
                assert math.isclose(value, float(row[key]), rel_tol=1e-12, abs_tol=1e-12)
    for band in ["early", "middle", "late", "all"]:
        values = snapshots[0]
        result["panels"][band] = dict(
            actual_tokens=sum(r["stats"][band]["tokens"] for r in values),
            contributing_prompts=sum(r["stats"][band]["tokens"] > 0 for r in values),
            domains={
                domain: sum(r["stats"][band]["tokens"] > 0 for r in values if r["domain"] == domain)
                for domain in sorted({r["domain"] for r in values})
            },
        )
    dump(destination / "coverage-audit.json", result)
    return result


def main():
    from examples.mirroropd.research.runtime.cuda import setup

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["bank", "score", "audit"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arm")
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    study = Study.load(args.plan)
    if args.action == "audit":
        audit(study, args.arm)
        return
    setup()
    if args.action == "bank":
        data.generate(extended(study), args.shard, groups=GROUPS)
    else:
        score(study, args.arm)


if __name__ == "__main__":
    main()
