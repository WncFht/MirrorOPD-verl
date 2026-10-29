#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Track a full single-pass transfer matrix without averaging changing state sets."""

import argparse
import collections
import csv
import json
import statistics as st
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.core.statistics import mean_ci as ci
from examples.mirroropd.research.core.statistics import ratio_ci
from examples.mirroropd.research.studies.cross_batch.protocol import LONG_PASS, validate_long_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    a = ap.parse_args()
    root = a.data
    validate_long_pass(read(root / "protocol.json"))
    bank = read(root / "bank.json")
    order = sorted({(r["batch_index"], r["group"]) for r in bank})
    groups = [g for _, g in order]
    metrics = ["rkl", "fkl_shared", "js", "tv", "hp", "hq", "overlap16", "common_p16", "common_q16"]
    summaries = []
    influences = []
    incoming = []
    retention = []
    fresh = []
    matrices = {}
    fit_retention = []
    for teacher in ["justrl", "polaris"]:
        dest = root / teacher
        files = sorted(dest.glob("fixed-*.json"))
        if not files:
            continue
        baseline = read(dest / "fixed-00.json")
        base = {r["id"]: r for r in baseline}
        previous = None
        prior_batch = None
        matrix = []
        completed = []
        for file in files:
            rows = read(file)
            batch = rows[0]["completed_batches"]
            index = {r["id"]: r for r in rows}
            assert len(rows) == LONG_PASS.examples and index.keys() == base.keys()
            panels = {g: [r for r in rows if r["group"] == g] for g in groups}
            panels.update(
                train_pool=[r for r in rows if r["split"] == "train"],
                held_math=[r for r in rows if r["split"] == "heldout" and r["domain"] == "math"],
                held_code=[r for r in rows if r["domain"] == "code"],
                held_if=[r for r in rows if r["domain"] == "if"],
            )
            for panel, rs in panels.items():
                item = dict(
                    teacher=teacher,
                    completed_batches=batch,
                    updates=batch * LONG_PASS.reuse,
                    panel=panel,
                    prompts=len(rs),
                )
                for metric in metrics:
                    initial = st.mean(base[r["id"]]["stats"]["all"][metric] / LONG_PASS.horizon for r in rs)
                    value = st.mean(r["stats"]["all"][metric] / LONG_PASS.horizon for r in rs)
                    item[metric] = value
                    item[metric + "_initial"] = initial
                    tokens = sum(r["length"] for r in rs)
                    item[metric + "_per_valid_token"] = sum(r["stats"]["all"][metric] for r in rs) / tokens
                    if metric == "rkl":
                        item["relative_drop"] = 1 - value / initial
                summaries.append(item)
            current_group_risks = {
                g: st.mean(r["stats"]["all"]["rkl"] / LONG_PASS.horizon for r in panels[g]) for g in groups
            }
            matrix.append([current_group_risks[g] for g in groups])
            completed.append(batch)
            if previous is not None:
                assert batch == prior_batch + 1
                active = chr(64 + batch)
                for group in groups:
                    ids = [r["id"] for r in panels[group]]
                    before = st.mean(previous[i]["stats"]["all"]["rkl"] / LONG_PASS.horizon for i in ids)
                    after = current_group_risks[group]
                    initial = st.mean(base[i]["stats"]["all"]["rkl"] / LONG_PASS.horizon for i in ids)
                    row = dict(
                        teacher=teacher,
                        training_batch=batch,
                        training_group=active,
                        evaluated_group=group,
                        before=before,
                        after=after,
                        absolute_drop=before - after,
                        drop_initial_pp=100 * (before - after) / initial,
                    )
                    influences.append(row)
                    if group in groups[: batch - 1]:
                        retention.append(row | dict(increase=after - before))
                train_before = st.mean(
                    previous[r["id"]]["stats"]["all"]["rkl"] / LONG_PASS.horizon for r in panels[active]
                )
                train_after = current_group_risks[active]
                train_initial = st.mean(
                    base[r["id"]]["stats"]["all"]["rkl"] / LONG_PASS.horizon for r in panels[active]
                )
                held_ids = [r["id"] for r in panels["held_math"]]
                held_before = st.mean(previous[i]["stats"]["all"]["rkl"] / LONG_PASS.horizon for i in held_ids)
                held_after = st.mean(index[i]["stats"]["all"]["rkl"] / LONG_PASS.horizon for i in held_ids)
                direct = train_before - train_after
                item = dict(
                    teacher=teacher,
                    training_batch=batch,
                    training_group=active,
                    initial_risk=train_initial,
                    risk_before_first_training=train_before,
                    risk_after_training=train_after,
                    forward_transfer_drop=1 - train_before / train_initial,
                    within_batch_drop=1 - train_after / train_before,
                    held_math_before=held_before,
                    held_math_after=held_after,
                    held_drop_per_local_drop=(held_before - held_after) / direct if direct > 0 else None,
                )
                for label, gs in [("previously_trained", groups[: batch - 1]), ("future_train", groups[batch:20])]:
                    ids = [r["id"] for r in rows if r["group"] in gs]
                    item[label + "_prompts"] = len(ids)
                    item[label + "_mean_drop"] = (
                        st.mean(
                            (previous[i]["stats"]["all"]["rkl"] - index[i]["stats"]["all"]["rkl"]) / LONG_PASS.horizon
                            for i in ids
                        )
                        if ids
                        else None
                    )
                incoming.append(item)
            previous = index
            prior_batch = batch
        matrices[teacher] = dict(groups=groups, completed_batches=completed, risk=matrix)
        for j in range(completed[-1]):
            before = matrix[j][j]
            after = matrix[j + 1][j]
            latest = matrix[-1][j]
            fit_retention.append(
                dict(
                    teacher=teacher,
                    group=groups[j],
                    trained_at_batch=j + 1,
                    evaluated_at_batch=completed[-1],
                    before_first_training=before,
                    immediately_after_training=after,
                    latest=latest,
                    latest_change_from_immediate=latest - after,
                    retained_local_gain=(before - latest) / (before - after) if before > after else None,
                )
            )
        with (root / f"{teacher}-risk-matrix.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["completed_batches", "optimizer_updates"] + groups)
            for batch, values in zip(completed, matrix, strict=False):
                writer.writerow([batch, LONG_PASS.reuse * batch] + values)
        baseline_fresh = dest / "fresh-00.json"
        if baseline_fresh.exists():
            bf = read(baseline_fresh)
            bfi = {(r["id"], r["seed"]): r for r in bf}
            for file in sorted(dest.glob("fresh-*.json")):
                rows = read(file)
                if len(rows) != 64:
                    continue
                delta = collections.defaultdict(list)
                for r in rows:
                    delta[r["id"]].append(r["stats"]["all"]["rkl"] - bfi[r["id"], r["seed"]]["stats"]["all"]["rkl"])
                values = [st.mean(v) for v in delta.values()]
                ids = sorted(delta)
                before_values = [st.mean(r["stats"]["all"]["rkl"] for r in bf if r["id"] == i) for i in ids]
                after_values = [st.mean(r["stats"]["all"]["rkl"] for r in rows if r["id"] == i) for i in ids]
                total = sum(r["stats"]["all"]["alive"] for r in rows)
                fresh.append(
                    dict(
                        teacher=teacher,
                        completed_batches=rows[0]["completed_batches"],
                        sequence_rkl=st.mean(r["stats"]["all"]["rkl"] for r in rows),
                        initial_sequence_rkl=st.mean(r["stats"]["all"]["rkl"] for r in bf),
                        difference=st.mean(values),
                        paired_prompt_ci=ci(values),
                        ratio_paired_prompt_ci=ratio_ci(before_values, after_values),
                        mean_length=st.mean(len(r["response_ids"]) for r in rows),
                        overlap16=sum(r["stats"]["all"]["overlap16"] for r in rows) / total,
                        common_p16=sum(r["stats"]["all"]["common_p16"] for r in rows) / total,
                    )
                )
    dump(root / "risk-summary.json", summaries)
    dump(root / "transfer-matrix.json", matrices)
    dump(root / "block-influence.json", influences)
    dump(root / "incoming-batches.json", incoming)
    dump(root / "retention.json", retention)
    dump(root / "fresh-summary.json", fresh)
    dump(root / "fit-retention.json", fit_retention)
    report = {}
    for teacher, matrix in matrices.items():
        last = matrix["completed_batches"][-1]
        panel = {r["panel"]: r for r in summaries if r["teacher"] == teacher and r["completed_batches"] == last}
        rec = [r for r in retention if r["teacher"] == teacher]
        report[teacher] = dict(
            completed_batches=last,
            updates=last * 16,
            risks={g: panel[g]["rkl"] for g in ["train_pool", "held_math", "held_code", "held_if"]},
            drops={g: panel[g]["relative_drop"] for g in ["train_pool", "held_math", "held_code", "held_if"]},
            old_batch_checks=len(rec),
            old_batch_increases_above_1e_6=sum(r["increase"] > 1e-6 for r in rec),
        )
    phases = []
    within = []
    cross = []
    for teacher, matrix in matrices.items():
        lookup = {(r["completed_batches"], r["panel"]): r for r in summaries if r["teacher"] == teacher}
        train = read(root / teacher / "training.json") if (root / teacher / "training.json").exists() else []
        probes = read(root / teacher / "update-probes.json") if (root / teacher / "update-probes.json").exists() else []
        for lo, hi in [(1, 5), (6, 10), (11, 15), (16, 20)]:
            if matrix["completed_batches"][-1] < hi:
                continue
            blocks = [r for r in incoming if r["teacher"] == teacher and lo <= r["training_batch"] <= hi]
            before = lookup[lo - 1, "held_math"]["rkl"]
            after = lookup[hi, "held_math"]["rkl"]
            local = sum(r["risk_before_first_training"] - r["risk_after_training"] for r in blocks)
            prs = [r for r in probes if lo <= r["training_batch"] <= hi]
            phases.append(
                dict(
                    teacher=teacher,
                    first_batch=lo,
                    last_batch=hi,
                    held_before=before,
                    held_after=after,
                    held_absolute_drop=before - after,
                    held_relative_drop=1 - after / before,
                    mean_local_relative_drop=st.mean(r["within_batch_drop"] for r in blocks),
                    held_drop_per_local_drop=(before - after) / local,
                    mean_raw_gradient_cosine=st.mean(r["raw_train_held_cosine"] for r in prs),
                    mean_actual_update_alignment=st.mean(r["descent_alignment"] for r in prs),
                    mean_held_gradient_norm=st.mean(r["held_gradient_norm"] for r in prs),
                    mean_parameter_step_norm=st.mean(r["parameter_step_norm"] for r in prs),
                )
            )
        for probe in probes:
            b = probe["training_batch"]
            file = root / teacher / f"fixed-{b:02d}.json"
            if not file.exists():
                continue
            rows = read(file)
            after = st.mean(
                r["stats"]["all"]["rkl"] / LONG_PASS.horizon for r in rows if r["group"] in ["M01", "M02", "M03", "M04"]
            )
            tr = [r for r in train if r["training_batch"] == b]
            local_after = lookup[b, chr(64 + b)]["rkl"]
            within.append(
                dict(
                    teacher=teacher,
                    batch=b,
                    probe_before=probe["held_before"],
                    probe_after_first=probe["held_after"],
                    probe_after_block=after,
                    probe_first_drop=probe["held_before"] - probe["held_after"],
                    probe_remaining15_drop=probe["held_after"] - after,
                    local_before=tr[0]["loss"],
                    local_after_first=tr[1]["loss"],
                    local_after_block=local_after,
                )
            )
        cf = root / f"cross-score-{teacher}.json"
        if cf.exists():
            cases = read(cf)
            if len(cases) == 4:
                case = {(r["parameters"], r["states"]): r for r in cases}
                indexes = {k: {(r["id"], r["seed"]): r for r in v["records"]} for k, v in case.items()}
                gain = collections.defaultdict(list)
                cost = collections.defaultdict(list)
                for key in indexes["initial", "initial"]:
                    l00 = indexes["initial", "initial"][key]["sums"]["rkl"]
                    l10 = indexes["final", "initial"][key]["sums"]["rkl"]
                    l11 = indexes["final", "final"][key]["sums"]["rkl"]
                    gain[key[0]].append(l00 - l10)
                    cost[key[0]].append(l11 - l10)
                gains = [st.mean(v) for v in gain.values()]
                costs = [st.mean(v) for v in cost.values()]
                means = {
                    f"{p}_{s}": case[p, s]["sequence_rkl"] for p in ["initial", "final"] for s in ["initial", "final"]
                }
                decode0 = next(
                    r["sequence_rkl"] for r in fresh if r["teacher"] == teacher and r["completed_batches"] == 0
                )
                decode1 = next(
                    r["sequence_rkl"] for r in fresh if r["teacher"] == teacher and r["completed_batches"] == 20
                )
                cross.append(
                    dict(
                        teacher=teacher,
                        means=means,
                        parameter_gain=st.mean(gains),
                        parameter_gain_ci=ci(gains),
                        visitation_cost=st.mean(costs),
                        visitation_cost_ci=ci(costs),
                        net_gain=means["initial_initial"] - means["final_final"],
                        diagonal_scoring_minus_decode=[
                            means["initial_initial"] - decode0,
                            means["final_final"] - decode1,
                        ],
                    )
                )
    dump(root / "phase-summary.json", phases)
    dump(root / "within-block-summary.json", within)
    dump(root / "cross-summary.json", cross)
    dump(root / "progress-summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
