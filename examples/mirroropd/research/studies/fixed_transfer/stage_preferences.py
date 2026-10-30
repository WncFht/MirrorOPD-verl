# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Tulu stage preference directions scored on identical frozen student states."""

import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from examples.mirroropd.research.backends import hf_models, hf_rkl
from examples.mirroropd.research.core.artifacts import complete, file_hash, receipt
from examples.mirroropd.research.core.io import dump, read, write_csv
from examples.mirroropd.research.core.preference_geometry import geometry, moments
from examples.mirroropd.research.runtime.cuda import setup
from examples.mirroropd.research.studies.fixed_transfer.analysis import PANELS
from examples.mirroropd.research.studies.fixed_transfer.data import required_rows
from examples.mirroropd.research.studies.fixed_transfer.spec import Study, fingerprint
from examples.mirroropd.research.studies.fixed_transfer.training import (
    arm_by_name,
    checkpoint_directory,
    scientific_inputs,
    teacher_state,
    validate_checkpoint,
)

STAGES = ("L1", "L2", "L3")
CONTRASTS = (("L1", "L2"), ("L2", "L3"))


def inputs(study):
    from examples.mirroropd.research.studies.fixed_transfer.runner import check_inputs

    arms = [arm_by_name(study, name) for name in STAGES]
    if len({a.updates for a in arms}) != 1:
        raise ValueError("Stage attribution requires equal update budgets")
    rows = required_rows(study, arms[0])
    for arm in arms[1:]:
        if fingerprint(required_rows(study, arm)) != fingerprint(rows):
            raise ValueError("Stage attribution requires identical frozen states")
    weights = {}
    for arm in arms:
        root = study.directory / "arms" / arm.name
        if read(root / "complete.json")["updates"] != arm.updates:
            raise ValueError("Stage endpoint is not complete")
        path = checkpoint_directory(study, arm, arm.updates)
        check_inputs(path / "complete.json", dict(scientific_inputs(study, arm), step=arm.updates))
        saved = validate_checkpoint(path, verify_hash=True)
        weights[arm.name] = dict(path=str(path), identity=saved["weight_identity"])
    return dict(
        study=study.identity,
        bank=fingerprint(rows),
        weights=weights,
        teachers={a.name: study.models[a.teacher].identity for a in arms},
        source=file_hash(__file__),
        geometry=file_hash(moments.__code__.co_filename),
        backend=file_hash(hf_models.__file__),
        horizon=study.training.horizon,
        weighting="fixed original student p0; full student vocabulary; sum/(prompts*H)",
        precision="FP32 student parameters/head; BF16 forward; FP64 preference reductions",
    ), rows


@torch.no_grad()
def cache_student(study, directory, identity, rows, path=None):
    cache_inputs = dict(identity=identity, bank=fingerprint(rows), backend=file_hash(hf_models.__file__), fp32=True)
    if complete(directory, cache_inputs):
        return
    model = hf_models.load(study.models[study.student], trainable=True, path=path).eval().requires_grad_(False)
    hf_rkl.save_tensor(directory / "head.pt", model.get_output_embeddings().weight.detach().cpu())
    for row in rows:
        hidden = hf_models.hidden(model, [row], pad_id=study.evaluation.stop_ids[0]).cpu()
        hf_rkl.save_tensor(directory / f"{row['id']:04d}.pt", hidden)
    receipt(directory, cache_inputs, prompts=len(rows))
    del model
    gc.collect()
    torch.cuda.empty_cache()


def summarize(records, horizon):
    panels = dict(PANELS)
    panels["train_all"] = list("ABCDEFGHIJKLMNOPQRST")
    panels["held_all"] = PANELS["held_math"] + PANELS["held_code"] + PANELS["held_if"]
    results, paired = [], []
    for panel, groups in panels.items():
        for a, b in CONTRASTS:
            stage_rows = {}
            for student in STAGES:
                chosen = sorted(
                    (
                        r
                        for r in records
                        if r["group"] in groups and r["a"] == a and r["b"] == b and r["student"] == student
                    ),
                    key=lambda r: r["id"],
                )
                if not chosen:
                    continue
                stage_rows[student] = chosen
                means = {
                    key: sum(r[key] for r in chosen) / (len(chosen) * horizon)
                    for key in ["d2", "u2", "du", "teacher_a_extra_mass", "teacher_b_extra_mass"]
                }
                results.append(
                    dict(
                        panel=panel,
                        a=a,
                        b=b,
                        student=student,
                        prompts=len(chosen),
                        actual_tokens=sum(r["length"] for r in chosen),
                        teacher_a_extra_mass_valid_mean=sum(r["teacher_a_extra_mass"] for r in chosen)
                        / sum(r["length"] for r in chosen),
                        teacher_b_extra_mass_valid_mean=sum(r["teacher_b_extra_mass"] for r in chosen)
                        / sum(r["length"] for r in chosen),
                        **means,
                        **geometry(**{k: means[k] for k in ["d2", "u2", "du"]}),
                    )
                )
            if not stage_rows:
                continue
            left, right = stage_rows[a], stage_rows[b]
            assert [r["id"] for r in left] == [r["id"] for r in right]
            denominator = np.array([r["d2"] for r in left])
            assert np.allclose(denominator, [r["d2"] for r in right], atol=1e-10, rtol=1e-10)
            numerator = np.array([y["du"] - x["du"] for x, y in zip(left, right, strict=True)])
            if denominator.mean() / horizon <= 1e-12:
                paired.append(dict(panel=panel, a=a, b=b, identifiable=False))
                continue
            rng = np.random.default_rng(42)
            sample = rng.integers(0, len(left), size=(5000, len(left)))
            den = denominator[sample].sum(1)
            valid = den / (len(left) * horizon) > 1e-12
            draws = numerator[sample][valid].sum(1) / den[valid]
            interval = np.quantile(draws, [0.025, 0.975]).tolist() if valid.all() else [None, None]
            paired.append(
                dict(
                    panel=panel,
                    a=a,
                    b=b,
                    identifiable=True,
                    later_minus_earlier_projection=float(numerator.sum() / denominator.sum()),
                    ci_low=interval[0],
                    ci_high=interval[1],
                    unstable_bootstrap_draws=int((~valid).sum()),
                    inference="paired prompt bootstrap; 5000 draws; seed42; one training seed",
                )
            )
    return results, paired


@torch.no_grad()
def score(study):
    identity, rows = inputs(study)
    output = study.directory / "analysis/stage-preferences"
    if complete(output, identity):
        return
    local = output / "student-hidden"
    cache_student(study, local / "before", study.models[study.student].identity, rows)
    for name, weight in identity["weights"].items():
        cache_student(study, local / name, weight["identity"], rows, weight["path"])
    teachers = {name: teacher_state(study, arm_by_name(study, name).teacher, rows) for name in STAGES}
    heads = {
        name: torch.load(local / name / "head.pt", map_location="cuda", weights_only=True).float()
        for name in ("before", *STAGES)
    }
    support = heads["before"].shape[0]
    records = []
    for row in rows:
        hidden = {
            name: torch.load(local / name / f"{row['id']:04d}.pt", map_location="cuda", weights_only=True)
            for name in heads
        }
        sums = {(a, b, name): {} for a, b in CONTRASTS for name in STAGES}
        for start in range(0, len(row["response_ids"]), 64):
            sl = slice(start, start + 64)
            lp = {
                name: torch.nn.functional.linear(h[sl].float(), heads[name]).log_softmax(-1)
                for name, h in hidden.items()
            }
            lq = {
                name: torch.nn.functional.linear(states[row["id"]][sl].to("cuda").float(), head).log_softmax(-1)[
                    :, :support
                ]
                for name, (head, states) in teachers.items()
            }
            for (a, b, name), total in sums.items():
                for key, value in moments(lp["before"], lq[a], lq[b], lp[name]).items():
                    total[key] = total.get(key, 0.0) + float(value.sum())
        for (a, b, name), total in sums.items():
            records.append(
                dict(
                    id=row["id"],
                    group=row["group"],
                    length=len(row["response_ids"]),
                    source=row["source"],
                    a=a,
                    b=b,
                    student=name,
                    **total,
                )
            )
        if len(records) % (6 * 32) == 0:
            print("STAGE_PREFERENCES", len(records) // 6, len(rows), flush=True)
    write_csv(output / "prompt-moments.csv", records)
    summary, paired = summarize(records, study.training.horizon)
    write_csv(output / "geometry.csv", summary)
    write_csv(output / "paired-stage-projection.csv", paired)
    dump(output / "inputs.json", identity)
    receipt(output, identity, prompts=len(rows), scientific_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    setup()
    score(Study.load(args.plan))


if __name__ == "__main__":
    main()
