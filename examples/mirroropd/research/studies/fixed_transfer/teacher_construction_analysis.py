# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Recompute teacher-construction readouts from identified, paired raw records.

Semantic ledger: one risk point is one arm, bank, domain and saved update.
Questions are the bootstrap unit; confidence intervals condition on one seed.
Candidate weights describe a data distribution, never trained-model accuracy.
Geometry ledger: bar plots start at zero; risk plots retain increases and use
actual optimizer updates. Method colors are fixed across panels. No fitted
convergence curve, learned predictor or capability matching is implied.
"""

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from examples.mirroropd.research.core.plotting import load_style, save_figure

DOMAINS = ["math", "code", "science", "if"]
ARM_HUES = {"RAW": "vcoral", "CCS-rho0": "vgray", "CCS-rho1": "vteal", "ERT-verifier": "vviolet"}
BENCHMARKS = ["MATH-500", "AMC23", "AIME24", "AIME25", "AIME26"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    """Strict readers are used on snapshots; do not certify a live partial file."""
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def identified(values, expected, label):
    """Reject duplicates and foreign identities even in a partial readout."""
    result = {}
    for value in values:
        key = value["id"]
        if key in result or key not in expected:
            raise ValueError(f"Duplicate/unexpected identity in {label}: {key}")
        if value.get("benchmark") != expected[key].get("benchmark"):
            raise ValueError(f"Benchmark identity mismatch in {label}: {key}")
        result[key] = value
    return result


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def table(path, values):
    if not values:
        raise ValueError(f"No complete readouts for {path.name}")
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def interval(values):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite bootstrap statistic")
    return np.quantile(values, [0.025, 0.975]).tolist()


def risk_statistics(before, after, boot_before, boot_after):
    initial, final = float(before), float(after)
    stable = initial > 1e-8 and min(boot_before) > 1e-8
    return dict(
        initial_rkl=initial,
        rkl=final,
        initial_rkl_ci95=interval(boot_before),
        rkl_ci95=interval(boot_after),
        absolute_drop=initial - final,
        absolute_drop_ci95=interval(boot_before - boot_after),
        residual_fraction=final / initial if stable else None,
        residual_ci95=interval(boot_after / boot_before) if stable else None,
        ratio_stable=bool(stable),
    )


def metric_tail_statistics(values, ids, seed=20261108):
    """Expose leverage without trimming or replacing the primary mean.

    Concentration summaries are defined only for nonnegative observations with
    positive total mass. Every question keeps its original evaluation weight.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != len(ids) or not len(values) or not np.isfinite(values).all():
        raise ValueError("Metric tail audit requires paired, nonempty finite values")
    largest = int(values.argmax())
    total = float(values.sum())
    nonnegative = bool(np.all(values >= 0))
    concentrated = nonnegative and total > 0
    indices = np.random.default_rng(seed).integers(0, len(values), (2000, len(values)))
    return dict(
        n=len(values),
        mean=float(values.mean()),
        mean_ci95=interval(values[indices].mean(1)),
        median=float(np.median(values)),
        p95=float(np.quantile(values, 0.95)),
        minimum=float(values.min()),
        maximum=float(values[largest]),
        largest_question_id=ids[largest],
        nonnegative=nonnegative,
        largest_question_fraction=float(values[largest] / total) if concentrated else None,
        effective_question_count=float(total**2 / np.square(values).sum()) if concentrated else None,
        leave_one_out_mean_min=float((total - values.max()) / (len(values) - 1)) if len(values) > 1 else None,
        leave_one_out_mean_max=float((total - values.min()) / (len(values) - 1)) if len(values) > 1 else None,
    )


def feature_question_table(root, partial=False):
    protocol = json.loads((root / "protocol.json").read_text())
    result = []
    for population, label, filename in [
        ("evaluation", "Base-bank", "heldout.jsonl"),
        ("evaluation", "teacher-bank", "heldout-teacher-bank.jsonl"),
        ("dev", "development", "development.jsonl"),
    ]:
        prompts = records(root / "prepared" / f"{population}.jsonl")
        expected = {d: {v["id"] for v in prompts if v["domain"] == d} for d in DOMAINS}
        for arm in protocol["arms"]:
            grouped = defaultdict(dict)
            for value in records(root / "opd" / arm / filename):
                if value["step"] != 0:
                    continue
                domain, key = value["domain"], value["id"]
                if domain not in expected or key not in expected[domain] or key in grouped[domain]:
                    raise ValueError("Duplicate/unexpected initial feature identity")
                grouped[domain][key] = value
            for domain in DOMAINS:
                ids = sorted(expected[domain])
                if set(grouped[domain]) != set(ids):
                    if not partial:
                        raise ValueError(f"Incomplete initial features: {arm}/{label}/{domain}")
                    continue
                for key in ids:
                    value = grouped[domain][key]
                    result.append(
                        dict(
                            arm=arm,
                            bank=label,
                            domain=domain,
                            id=key,
                            tokens=value.get("tokens"),
                            **{k: value[k] for k in ["rkl", "fkl", "chi2", "centered_logratio_variance"]},
                        )
                    )
    return result


def feature_tail_table(points):
    grouped = defaultdict(list)
    for value in points:
        grouped[value["arm"], value["bank"], value["domain"]].append(value)
    result = []
    for (arm, bank, domain), values in sorted(grouped.items()):
        ids = [v["id"] for v in values]
        for metric in ["rkl", "fkl", "chi2", "centered_logratio_variance"]:
            audit = metric_tail_statistics([v[metric] for v in values], ids, seed=20261108 + DOMAINS.index(domain))
            lengths = [v.get("tokens") for v in values]
            known_lengths = all(isinstance(v, int) and v > 0 for v in lengths)
            scores = np.array([v[metric] for v in values])
            # These secondary quantities expose the fixed measure's sensitivity;
            # they never replace the protocol's equal-question risk or its CI.
            audit.update(
                largest_question_response_tokens=lengths[int(scores.argmax())],
                token_weighted_mean_diagnostic=float(np.average(scores, weights=lengths)) if known_lengths else None,
                short_responses_count_le16=sum(v <= 16 for v in lengths) if known_lengths else None,
                short_responses_metric_fraction=(
                    float(scores[np.array(lengths) <= 16].sum() / scores.sum())
                    if known_lengths and audit["nonnegative"] and scores.sum() > 0
                    else None
                ),
            )
            result.append(
                dict(
                    arm=arm,
                    bank=bank,
                    domain=domain,
                    metric=metric,
                    **audit,
                )
            )
    return result


def initial_metric_figure(points, output):
    """Plot the actual per-question initial distances; do not fit a mapping."""
    import matplotlib.pyplot as plt

    selected = [v for v in points if v["bank"] == "development"]
    if not selected:
        raise ValueError("No complete development initial feature bank")
    fv = load_style()
    theme = fv.apply_rc("paper", fontsize=8.5)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.19, top=0.88, hspace=0.6, wspace=0.4)
    handles = {}
    for ax, domain in zip(axes.flat, DOMAINS, strict=True):
        for arm, hue in ARM_HUES.items():
            values = [v for v in selected if v["arm"] == arm and v["domain"] == domain]
            if not values:
                continue
            artist = ax.scatter(
                [v["rkl"] for v in values], [v["chi2"] for v in values], s=9, alpha=0.6, color=fv.HUES[hue], label=arm
            )
            handles[arm] = artist
        ax.set(title=domain, xlabel="Initial RKL (nats/token)", ylabel="Initial Pearson chi-square")
        for coord, metric in [("x", "rkl"), ("y", "chi2")]:
            data = np.array([v[metric] for v in selected])
            scale = "log" if np.all(data > 0) else "symlog"
            set_scale = ax.set_xscale if coord == "x" else ax.set_yscale
            set_limits = ax.set_xlim if coord == "x" else ax.set_ylim
            set_scale(scale, **({"linthresh": 1e-7} if scale == "symlog" else {}))
            if scale == "log":
                set_limits(data.min() / 1.8, data.max() * 1.8)
            else:
                set_limits(min(0.0, data.min()) * 1.2 - 1e-7, max(1e-7, data.max()) * 1.8)
        fv.apply_frame(ax, theme.frame)
        fv.apply_grid(ax, theme.grid)
    fig.legend(list(handles.values()), list(handles), loc="upper center", ncol=4, frameon=False)
    save_figure(
        fig,
        output / "initial-distance-scatter",
        "Independent development bank | step 0 | per-question token mean | no fitted relation",
        caption_size=7,
    )


def capability_statistics(base, teacher, student, seed=20261108, resamples=2000):
    values = np.array([base, teacher, student], dtype=float)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("Capability records must be three nonempty, paired finite arrays")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Capability scores must be in [0,1]")
    indices = np.random.default_rng(seed).integers(0, values.shape[1], (resamples, values.shape[1]))
    boots = values[:, indices].mean(-1)
    means = values.mean(-1)
    gains = boots[1] - boots[0]
    gain_ci = interval(gains)
    # Conservatively suppress recovery ratios when any paired replicate has a
    # nonpositive denominator; never discard inconvenient bootstrap replicates.
    stable = gain_ci[0] > 0 and np.all(gains > 0)
    return dict(
        base=float(means[0]),
        base_ci95=interval(boots[0]),
        teacher=float(means[1]),
        teacher_ci95=interval(boots[1]),
        student=float(means[2]),
        student_ci95=interval(boots[2]),
        teacher_gain=float(means[1] - means[0]),
        teacher_gain_ci95=gain_ci,
        student_gain=float(means[2] - means[0]),
        student_gain_ci95=interval(boots[2] - boots[0]),
        student_minus_teacher=float(means[2] - means[1]),
        student_minus_teacher_ci95=interval(boots[2] - boots[1]),
        gain_recovered=float((means[2] - means[0]) / (means[1] - means[0])) if stable else None,
        gain_recovered_ci95=interval((boots[2] - boots[0]) / gains) if stable else None,
        recovery_denominator_stable=bool(stable),
    )


def token_costs(teacher, student, ert=False):
    if [r["step"] for r in teacher] != list(range(1, len(teacher) + 1)):
        raise ValueError("Noncontiguous teacher updates")
    if [r["step"] for r in student] != list(range(1, len(student) + 1)):
        raise ValueError("Noncontiguous OPD updates")
    outer = {}
    for value in student:
        if value["outer"] < 0 or value["sampled_tokens"] < 0:
            raise ValueError("Invalid rollout cost")
        previous = outer.setdefault(value["outer"], value["sampled_tokens"])
        if previous != value["sampled_tokens"]:
            raise ValueError("Different reuse steps claim different fresh rollout lengths")
    return dict(
        teacher_updates=len(teacher),
        teacher_response_token_presentations=(
            sum(value["sampled_tokens"] for value in teacher)
            if ert
            else teacher[-1]["presented_tokens"]
            if teacher
            else 0
        ),
        opd_updates=len(student),
        opd_fresh_response_tokens=sum(outer.values()),
        opd_response_token_presentations=sum(value["sampled_tokens"] for value in student),
    )


def capability_tables(root, partial=False):
    protocol = json.loads((root / "protocol.json").read_text())
    expected = {v["id"]: v for v in records(root / "prepared/benchmark.jsonl")}
    if len(expected) != protocol["populations"]["benchmark"]["count"]:
        raise ValueError("Frozen benchmark population mismatch")
    groups = sorted({v["benchmark"] for v in expected.values()})
    output, behavior = [], []
    base = identified(records(root / "benchmarks/base/responses.jsonl"), expected, "Base")
    for arm in protocol["arms"]:
        paired = [base]
        for label in ["teacher", "student"]:
            path = root / "benchmarks" / f"{arm}-{label}/responses.jsonl"
            paired.append(identified(records(path), expected, f"{arm}/{label}"))
        for benchmark in groups:
            ids = sorted(k for k, v in expected.items() if v["benchmark"] == benchmark)
            if any(not set(ids).issubset(v) for v in paired):
                if not partial:
                    raise ValueError(f"Incomplete benchmark {arm}/{benchmark}")
                continue
            scores = [[v[k]["reward"] for k in ids] for v in paired]
            output.append(dict(arm=arm, benchmark=benchmark, n=len(ids), **capability_statistics(*scores)))
            for label, values in zip(["base", "teacher", "student"], paired, strict=True):
                selected = [values[k] for k in ids]
                behavior.append(
                    dict(
                        arm=arm,
                        checkpoint=label,
                        benchmark=benchmark,
                        n=len(ids),
                        response_tokens_mean=float(np.mean([v["tokens"] for v in selected])),
                        response_tokens_median=float(np.median([v["tokens"] for v in selected])),
                        length_stop_fraction=float(np.mean([v["finish_reason"] == "length" for v in selected])),
                        canonical_accuracy=float(np.mean([v["reward"] for v in selected])),
                        dapo_accuracy=float(np.mean([v["dapo_verifier_reward"] for v in selected])),
                        grader_disagreement=float(
                            np.mean([v["reward"] != v["dapo_verifier_reward"] for v in selected])
                        ),
                    )
                )
    return output, behavior


def capability_figures(values, output, fixture=False):
    """Show paired absolute gains; unstable recovery ratios are never plotted."""
    import matplotlib.pyplot as plt

    if not values:
        return False
    indexed = {(v["arm"], v["benchmark"]): v for v in values}
    fv = load_style()
    theme = fv.apply_rc("paper", fontsize=8)
    bounds = [v[f"{stage}_gain_ci95"] for v in values for stage in ["teacher", "student"]]
    low, high = min(0.0, min(v[0] for v in bounds)) * 100, max(0.0, max(v[1] for v in bounds)) * 100
    pad = max(1.0, (high - low) * 0.12)
    fig, axes = plt.subplots(2, 5, figsize=(11.5, 5.3))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.2, top=0.88, wspace=0.55, hspace=0.65)
    handles = {}
    for row, stage in enumerate(["teacher", "student"]):
        for col, benchmark in enumerate(BENCHMARKS):
            ax = axes[row, col]
            for index, (arm, hue) in enumerate(ARM_HUES.items()):
                if (arm, benchmark) not in indexed:
                    continue
                value = indexed[arm, benchmark]
                y = value[f"{stage}_gain"] * 100
                lo, hi = np.array(value[f"{stage}_gain_ci95"]) * 100
                ax.vlines(index, lo, hi, color=fv.HUES[hue], linewidth=1)
                ax.hlines([lo, hi], index - 0.055, index + 0.055, color=fv.HUES[hue], linewidth=1)
                handles[arm] = ax.scatter(index, y, s=17, color=fv.HUES[hue], label=arm)
            ax.axhline(0, color=fv.HUES["vgray"], linestyle=":", linewidth=0.6)
            ax.set(
                title=f"{stage.title()}: {benchmark}",
                xlabel="Construction",
                ylabel="Gain over Base (pp)",
                xticks=range(4),
                xticklabels=["RAW", "C0", "C1", "ERT"],
                xlim=(-0.5, 3.5),
                ylim=(low - pad, high + pad),
            )
            if not any((arm, benchmark) in indexed for arm in ARM_HUES):
                ax.text(0.5, 0.5, "No paired endpoint", transform=ax.transAxes, ha="center", va="center", fontsize=7)
            fv.apply_frame(ax, theme.frame)
            fv.apply_grid(ax, theme.grid)
    fig.legend(list(handles.values()), list(handles), loc="upper center", ncol=4, frameon=False)
    partial = len(indexed) != len(ARM_HUES) * len(BENCHMARKS)
    prefix = "LAYOUT FIXTURE - NOT EXPERIMENTAL DATA | " if fixture else "Partial endpoints | " if partial else ""
    save_figure(
        fig,
        output / "capability-gains",
        prefix + "Canonical grader | greedy cap 2048 | paired question-bootstrap 95% CI | one training seed",
        caption_size=7,
    )
    return True


def teacher_distance_figure(risks, capabilities, output, fixture=False):
    """Display observed capability/distance tradeoffs without fitting a frontier."""
    import matplotlib.pyplot as plt

    paired = []
    for arm in ARM_HUES:
        score = next((v for v in capabilities if v["arm"] == arm and v["benchmark"] == "MATH-500"), None)
        if score is None:
            continue
        distances = {v["domain"]: v for v in risks if v["arm"] == arm and v["bank"] == "Base-bank" and v["step"] == 0}
        if {"math", "macro4"}.issubset(distances):
            paired.append((arm, score, distances))
    if not paired:
        return False
    fv = load_style()
    theme = fv.apply_rc("paper", fontsize=8.5)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.25, top=0.82, wspace=0.5)
    ylow = min(0.0, min(v[1]["teacher_gain_ci95"][0] for v in paired)) * 100
    yhigh = max(0.0, max(v[1]["teacher_gain_ci95"][1] for v in paired)) * 100
    ypad = max(1.0, (yhigh - ylow) * 0.12)
    xbounds = [v[2][d]["rkl_ci95"] for v in paired for d in ["math", "macro4"]]
    positive = all(v[0] > 0 for v in xbounds)
    xlow, xhigh = min(v[0] for v in xbounds), max(v[1] for v in xbounds)
    handles = {}
    for ax, domain in zip(axes, ["math", "macro4"], strict=True):
        for arm, score, distances in paired:
            color = fv.HUES[ARM_HUES[arm]]
            distance = distances[domain]
            x, y = distance["rkl"], score["teacher_gain"] * 100
            ax.hlines(y, *distance["rkl_ci95"], color=color, linewidth=1)
            ax.vlines(x, *(np.array(score["teacher_gain_ci95"]) * 100), color=color, linewidth=1)
            handles[arm] = ax.scatter(x, y, s=20, color=color, label=arm)
        ax.set_xscale("log" if positive else "linear")
        ax.set_xlim(xlow / 1.8 if positive else min(0.0, xlow) - 1e-5, max(1e-5, xhigh) * 1.8)
        if positive:
            ticks = np.geomspace(*ax.get_xlim(), num=4)
            ax.set_xticks(ticks, labels=[f"{tick:.2g}" for tick in ticks])
            ax.minorticks_off()
        ax.set_ylim(ylow - ypad, yhigh + ypad)
        ax.axhline(0, color=fv.HUES["vgray"], linestyle=":", linewidth=0.6)
        ax.set(
            title=f"Initial Shared-Base: {domain}",
            xlabel="Initial RKL (nats/token)",
            ylabel="Teacher MATH-500 gain (pp)",
        )
        fv.apply_frame(ax, theme.frame)
        fv.apply_grid(ax, theme.grid)
    fig.legend(list(handles.values()), list(handles), loc="upper center", ncol=4, frameon=False)
    prefix = (
        "LAYOUT FIXTURE - NOT EXPERIMENTAL DATA\n" if fixture else "Partial endpoints | " if len(paired) < 4 else ""
    )
    save_figure(
        fig,
        output / "teacher-distance-capability",
        prefix + "Separate evaluation populations | pointwise 95% CI | no fitted frontier",
        caption_size=7,
    )
    return True


def cost_table(root, partial=False):
    protocol = json.loads((root / "protocol.json").read_text())
    result = []
    for arm in protocol["arms"]:
        teacher = records(root / "teachers" / arm / "metrics.jsonl")
        student = records(root / "opd" / arm / "metrics.jsonl")
        expected_teacher = protocol["ert" if arm == "ERT-verifier" else "sft"]["updates"]
        expected_student = protocol["opd"]["updates"]
        if not partial and (len(teacher) != expected_teacher or len(student) != expected_student):
            raise ValueError(f"Incomplete update costs {arm}")
        for value in student:
            if (value["outer"], value["reuse"]) != divmod(value["step"] - 1, protocol["opd"]["reuse"]):
                raise ValueError(f"Unexpected rollout reuse identity: {arm}/{value['step']}")
        result.append(
            dict(
                arm=arm,
                complete=len(teacher) == expected_teacher and len(student) == expected_student,
                **token_costs(teacher, student, ert=arm == "ERT-verifier"),
                teacher_update_loop_seconds=float(teacher[-1]["elapsed"]) if teacher else None,
                opd_update_loop_seconds=float(student[-1]["elapsed"]) if student else None,
            )
        )
    return result


def generation_costs(root, partial=False):
    """Count a shared candidate pool once, separately from learner token costs."""
    protocol = json.loads((root / "protocol.json").read_text())
    result = []
    for population, kind in [("teacher", "base"), ("teacher", "expert"), ("dev", "base"), ("evaluation", "base")]:
        expected = {v["id"] for v in records(root / "prepared" / f"{population}.jsonl")}
        if len(expected) != protocol["populations"][population]["count"]:
            raise ValueError(f"Generation population mismatch: {population}")
        seen, rewards, tokens, lengths, question_success = set(), [], 0, 0, 0
        response_count, fully_graded_questions = 0, 0
        k = protocol["generation"]["candidates"] if population == "teacher" and kind == "base" else 1
        for path in sorted((root / "generated").glob(f"{population}-{kind}-*.jsonl")):
            # Generated token-ID arrays can be large; stream immutable snapshots.
            with path.open() as stream:
                for line in stream:
                    value = json.loads(line)
                    if value["id"] in seen or value["id"] not in expected:
                        raise ValueError(f"Duplicate/unexpected generation identity: {population}/{kind}")
                    seen.add(value["id"])
                    candidates = value["candidates"]
                    if len(candidates) != k:
                        raise ValueError("Wrong generation candidate count")
                    question_success += any(v["reward"] is not None and v["reward"] > 0 for v in candidates)
                    fully_graded_questions += all(v["reward"] is not None for v in candidates)
                    for candidate in candidates:
                        reward = candidate["reward"]
                        if reward is not None and (not np.isfinite(reward) or not 0 <= reward <= 1):
                            raise ValueError("Invalid generation reward")
                        if candidate["tokens"] != len(candidate["generated_ids"]):
                            raise ValueError("Generation token count mismatch")
                        if reward is not None:
                            rewards.append(reward)
                        response_count += 1
                        tokens += candidate["tokens"]
                        lengths += candidate["finish_reason"] == "length"
        if seen != expected and not partial:
            raise ValueError(f"Incomplete generation costs: {population}/{kind}")
        result.append(
            dict(
                population=population,
                generator=kind,
                complete=seen == expected,
                questions=len(seen),
                responses=response_count,
                graded_responses=len(rewards),
                fully_graded_questions=fully_graded_questions,
                generated_response_tokens=tokens,
                candidate_mean_reward=float(np.mean(rewards)) if rewards else None,
                questions_with_positive_candidate=question_success,
                length_stop_fraction=lengths / response_count if response_count else None,
                shared_by="CCS-rho0,CCS-rho1" if population == "teacher" and kind == "base" else "",
            )
        )
    return result


def prediction_table(risks, features, endpoint=256):
    """Independent development features joined to permanent test endpoints.

    These rows are descriptive contrasts, not independent training replicates.
    No regression, target-selected feature choice or predictive R-squared is fit.
    """
    dev = {(v["arm"], v["domain"], v["step"]): v for v in risks if v["bank"] == "development"}
    initial = {(v["arm"], v["domain"]): v for v in features if v["bank"] == "development"}
    result = []
    for value in risks:
        if value["bank"] == "development" or value["step"] != endpoint:
            continue
        key = value["arm"], value["domain"]
        if key not in initial or any((*key, step) not in dev for step in [4, 16]):
            continue
        result.append(
            dict(
                arm=value["arm"],
                domain=value["domain"],
                test_bank=value["bank"],
                test_step=endpoint,
                **{f"dev_initial_{k}": initial[key][k] for k in ["rkl", "fkl", "chi2", "centered_logratio_variance"]},
                **{
                    f"dev_step{step}_{k}": dev[(*key, step)][k]
                    for step in [4, 16]
                    for k in ["absolute_drop", "residual_fraction"]
                },
                **{f"test_{k}": value[k] for k in ["initial_rkl", "rkl", "absolute_drop", "residual_fraction"]},
            )
        )
    return result


def method_contrasts(root):
    """Pair arms by question; own-teacher contrasts pair prompts, not prefixes.

    Intervals are pointwise descriptive contrasts without multiplicity control.
    Complete domains are required; missing endpoints never create a contrast.
    """
    protocol = json.loads((root / "protocol.json").read_text())
    prompts = records(root / "prepared/evaluation.jsonl")
    expected = {d: sorted(v["id"] for v in prompts if v["domain"] == d) for d in DOMAINS}
    result = []
    for bank, filename in [("Base-bank", "heldout.jsonl"), ("teacher-bank", "heldout-teacher-bank.jsonl")]:
        grouped = defaultdict(dict)
        for arm in protocol["arms"]:
            for value in records(root / "opd" / arm / filename):
                key = arm, value["domain"], value["step"]
                if value["id"] in grouped[key] or value["id"] not in expected[value["domain"]]:
                    raise ValueError("Duplicate/unexpected contrast identity")
                grouped[key][value["id"]] = value["rkl"]
        for left, right in combinations(protocol["arms"], 2):
            for step in protocol["evaluation"]["checkpoints"]:
                if step == 0:
                    continue
                domain_values = []
                for index, domain in enumerate(DOMAINS):
                    ids = expected[domain]
                    sets = [grouped[arm, domain, s] for arm, s in [(left, 0), (left, step), (right, 0), (right, step)]]
                    if any(set(v) != set(ids) for v in sets):
                        continue
                    values = np.array([[v[k] for k in ids] for v in sets])
                    differences = np.array([values[1] - values[3], (values[0] - values[1]) - (values[2] - values[3])])
                    indices = np.random.default_rng(20261108 + index).integers(0, len(ids), (2000, len(ids)))
                    means, boots = differences.mean(1), differences[:, indices].mean(-1)
                    domain_values.append((means, boots))
                    result.append(contrast_row(left, right, bank, domain, step, len(ids), means, boots))
                if len(domain_values) == len(DOMAINS):
                    means = np.mean([v[0] for v in domain_values], axis=0)
                    boots = np.mean([v[1] for v in domain_values], axis=0)
                    result.append(contrast_row(left, right, bank, "macro4", step, len(prompts), means, boots))
    return result


def contrast_row(left, right, bank, domain, step, n, means, boots):
    return dict(
        left=left,
        right=right,
        bank=bank,
        domain=domain,
        step=step,
        n=n,
        prefixes_matched=bank == "Base-bank",
        residual_rkl_left_minus_right=float(means[0]),
        residual_rkl_difference_ci95=interval(boots[0]),
        absolute_drop_left_minus_right=float(means[1]),
        absolute_drop_difference_ci95=interval(boots[1]),
    )


def risk_figures(values, output):
    """Render complete observed points only; preserve increases and uncertainty."""
    import matplotlib.pyplot as plt

    fv = load_style()
    theme = fv.apply_rc("paper", fontsize=8)
    for field, ci, ylabel, suffix in [
        ("rkl", "rkl_ci95", "RKL (nats/token)", "absolute"),
        ("residual_fraction", "residual_ci95", "Residual / initial RKL", "relative"),
    ]:
        complete_points = [v for v in values if v["bank"] != "development" and v[ci] is not None]
        extent = max(16, max((v["step"] for v in complete_points), default=0))
        ticks = [step for step in [0, 16, 64, 128, 256] if step <= extent and (step != 16 or extent <= 64)]
        low = min([0.0, *(v[ci][0] for v in complete_points)])
        high = max([1.0 if suffix == "relative" else 1e-8, *(v[ci][1] for v in complete_points)])
        fig, axes = plt.subplots(2, 5, figsize=(11.5, 5.2), sharex=True)
        fig.subplots_adjust(left=0.075, right=0.98, bottom=0.2, top=0.88, wspace=0.55, hspace=0.65)
        handles = {}
        for row, bank in enumerate(["Base-bank", "teacher-bank"]):
            for col, domain in enumerate([*DOMAINS, "macro4"]):
                ax = axes[row, col]
                points = [v for v in values if v["bank"] == bank and v["domain"] == domain]
                for arm, hue in ARM_HUES.items():
                    selected = sorted(
                        [v for v in points if v["arm"] == arm and v[field] is not None], key=lambda v: v["step"]
                    )
                    if not selected:
                        continue
                    x = [v["step"] for v in selected]
                    y = np.array([v[field] for v in selected])
                    bounds = np.array([v[ci] for v in selected])
                    artist = ax.errorbar(
                        x,
                        y,
                        yerr=[y - bounds[:, 0], bounds[:, 1] - y],
                        color=fv.HUES[hue],
                        marker="o",
                        markersize=3,
                        linewidth=1,
                        label=arm,
                    )
                    handles[arm] = artist
                ax.set(title=f"{bank}: {domain}", xlabel="OPD updates", ylabel=ylabel, xticks=ticks)
                ax.set_xlim(-extent * 0.04, extent * 1.04)
                ax.set_ylim(low - (high - low) * 0.04, high + (high - low) * 0.1)
                if not points:
                    ax.text(0.5, 0.5, "No complete readout", transform=ax.transAxes, ha="center", va="center")
                if field == "residual_fraction":
                    ax.axhline(1, color=fv.HUES["vgray"], linewidth=0.6, linestyle=":")
                fv.apply_frame(ax, theme.frame)
                fv.apply_grid(ax, theme.grid)
        if handles:
            fig.legend(list(handles.values()), list(handles), loc="upper center", ncol=4, frameon=False)
        save_figure(
            fig,
            output / f"absorption-{suffix}",
            "Observed checkpoints only | fixed prefix banks | paired question-bootstrap 95% CI | one training seed",
            caption_size=7,
        )


def snapshot_inputs(root, output, paths, partial=False):
    """Analyze immutable copies, recording live trailing fragments explicitly."""
    target = output / "inputs"
    receipt = {}
    for path in paths:
        if not path.exists():
            continue
        raw = path.read_bytes()
        omitted = 0
        if path.suffix == ".jsonl" and raw and not raw.endswith(b"\n"):
            if not partial:
                raise ValueError(f"Unterminated input line: {path}")
            end = raw.rfind(b"\n") + 1
            omitted, raw = len(raw) - end, raw[:end]
        destination = target / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        receipt[str(path.relative_to(root))] = dict(sha256=digest(destination), omitted_trailing_bytes=omitted)
    shutil.copyfile(__file__, output / "source.py")
    return target, receipt


def risk_table(root, partial=False):
    protocol = json.loads((root / "protocol.json").read_text())
    result = []
    features = []
    for arm in protocol["arms"]:
        for label, filename, population in [
            ("Base-bank", "heldout.jsonl", "evaluation"),
            ("teacher-bank", "heldout-teacher-bank.jsonl", "evaluation"),
            ("development", "development.jsonl", "dev"),
        ]:
            prompts = records(root / "prepared" / f"{population}.jsonl")
            expected = {d: {v["id"] for v in prompts if v["domain"] == d} for d in DOMAINS}
            grouped = defaultdict(dict)
            for value in records(root / "opd" / arm / filename):
                key = value["domain"], value["step"]
                if value["domain"] not in expected:
                    raise ValueError(f"Unexpected evaluation domain in {arm}/{filename}")
                if value["id"] in grouped[key] or value["id"] not in expected[value["domain"]]:
                    raise ValueError(f"Duplicate/unexpected identity in {arm}/{filename}/{key}")
                grouped[key][value["id"]] = value
            steps = sorted(set(protocol["evaluation"]["checkpoints"] + ([4] if population == "dev" else [])))
            if any(step not in steps for _, step in grouped):
                raise ValueError(f"Unexpected evaluation step in {arm}/{filename}")
            complete = defaultdict(list)
            for domain in DOMAINS:
                ids = sorted(expected[domain])
                indices = np.random.default_rng(20261108 + DOMAINS.index(domain)).integers(
                    0, len(ids), (2000, len(ids))
                )
                initial = grouped.get((domain, 0), {})
                for step in steps:
                    current = grouped.get((domain, step), {})
                    if set(current) != set(ids) or set(initial) != set(ids):
                        if not partial:
                            raise ValueError(f"Incomplete {arm}/{filename}/{domain}/{step}")
                        continue
                    before = np.array([initial[key]["rkl"] for key in ids])
                    after = np.array([current[key]["rkl"] for key in ids])
                    if not np.isfinite(before).all() or not np.isfinite(after).all():
                        raise ValueError("Nonfinite RKL input")
                    bb, ba = before[indices].mean(1), after[indices].mean(1)
                    complete[step].append((before.mean(), after.mean(), bb, ba))
                    result.append(
                        dict(
                            arm=arm,
                            bank=label,
                            domain=domain,
                            step=step,
                            n=len(ids),
                            **risk_statistics(before.mean(), after.mean(), bb, ba),
                        )
                    )
                if set(initial) == set(ids):
                    features.append(
                        dict(
                            arm=arm,
                            bank=label,
                            domain=domain,
                            **{
                                name: float(np.mean([v[name] for v in initial.values()]))
                                for name in ["rkl", "fkl", "chi2", "centered_logratio_variance"]
                            },
                        )
                    )
            for step, values in complete.items():
                if len(values) == len(DOMAINS):
                    before, after = np.mean([v[0] for v in values]), np.mean([v[1] for v in values])
                    bb, ba = np.mean([v[2] for v in values], axis=0), np.mean([v[3] for v in values], axis=0)
                    result.append(
                        dict(
                            arm=arm,
                            bank=label,
                            domain="macro4",
                            step=step,
                            n=len(prompts),
                            **risk_statistics(before, after, bb, ba),
                        )
                    )
            domain_features = [v for v in features if v["arm"] == arm and v["bank"] == label]
            if {v["domain"] for v in domain_features} == set(DOMAINS):
                features.append(
                    dict(
                        arm=arm,
                        bank=label,
                        domain="macro4",
                        **{
                            name: float(np.mean([v[name] for v in domain_features]))
                            for name in ["rkl", "fkl", "chi2", "centered_logratio_variance"]
                        },
                    )
                )
    return result, features


def candidate_figure(root, output):
    import matplotlib.pyplot as plt

    values = records(root / "teachers/CCS-rho1/candidate-weights.jsonl")
    prompts = records(root / "prepared/teacher.jsonl")
    if len(values) != len(prompts) or {v["id"] for v in values} != {v["id"] for v in prompts}:
        raise ValueError("Candidate weights do not cover the frozen training population")
    counts = Counter()
    uniform, weighted = [], []
    for value in values:
        rewards, weights = np.array(value["rewards"]), np.array(value["weights"])
        if len(rewards) != 8 or not np.isfinite(weights).all() or abs(weights.mean() - 1) > 1e-8:
            raise ValueError("Invalid K8 candidate weights")
        if min(weights) < 0.05 - 1e-8 or max(weights) > 4 + 1e-8 or np.mean((weights - 1) ** 2) > 1 + 1e-8:
            raise ValueError("Candidate weights violate frozen Pearson/box constraints")
        counts[int(rewards.sum())] += 1
        uniform.append(rewards.mean())
        weighted.append(np.mean(rewards * weights))
    saved = json.loads((root / "candidate-weight-expectations.json").read_text())
    if not np.isclose(np.mean(weighted), saved["ccs_rho1_pool_expected_reward"], atol=1e-12, rtol=0):
        raise ValueError("Recomputed candidate expectation disagrees with saved arithmetic")
    coverage = [dict(correct_count=k, questions=counts[k], fraction=counts[k] / len(values)) for k in range(9)]
    quality = [
        dict(arm=a, expected_reward=float(np.mean(v))) for a, v in [("CCS-rho0", uniform), ("CCS-rho1", weighted)]
    ]
    table(output / "candidate-coverage.csv", coverage)
    table(output / "candidate-reward.csv", quality)
    fv = load_style()
    theme = fv.apply_rc("paper", fontsize=8.5)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.27, top=0.79, wspace=0.62)
    axes[0].bar(range(9), [counts[k] for k in range(9)], color=fv.HUES["vgray"], width=0.75)
    axes[0].set(title="(a) Candidate coverage", xlabel="Correct count (K=8)", ylabel="Questions", xticks=range(9))
    axes[0].set_ylim(0, max(counts.values()) * 1.15)
    axes[1].bar(
        [0, 1],
        [v["expected_reward"] * 100 for v in quality],
        color=[fv.HUES[ARM_HUES[v["arm"]]] for v in quality],
        width=0.55,
    )
    axes[1].set(
        title="(b) Selected data reward",
        xlabel="Sampling rule",
        ylabel="Expected reward (%)",
        xticks=[0, 1],
        xticklabels=["Uniform", "CCS"],
    )
    axes[1].set_ylim(0, max(v["expected_reward"] * 100 for v in quality) * 1.3)
    for ax in axes:
        fv.apply_frame(ax, theme.frame)
        fv.apply_grid(ax, theme.grid)
    save_figure(
        fig,
        output / "candidate-data",
        "2,048 questions | same K=8 pool | data expectation, not trained accuracy",
        caption_size=7,
    )
    return dict(coverage=coverage, quality=quality)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--view", choices=["candidates", "features", "risks", "capabilities", "costs", "final"], required=True
    )
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = [root / "protocol.json"]
    if args.view == "candidates":
        inputs += [
            root / "teachers/CCS-rho1/candidate-weights.jsonl",
            root / "prepared/teacher.jsonl",
            root / "candidate-weight-expectations.json",
        ]
    if args.view in ["features", "risks", "final"]:
        inputs += [
            *sorted((root / "opd").glob("*/*heldout*.jsonl")),
            *sorted((root / "opd").glob("*/development.jsonl")),
            root / "prepared/evaluation.jsonl",
            root / "prepared/dev.jsonl",
        ]
    if args.view in ["capabilities", "final"]:
        inputs += [root / "prepared/benchmark.jsonl", *sorted((root / "benchmarks").glob("*/responses.jsonl"))]
    if args.view in ["costs", "final"]:
        inputs += [
            *sorted((root / "teachers").glob("*/metrics.jsonl")),
            *sorted((root / "opd").glob("*/metrics.jsonl")),
            *sorted((root / "generated").glob("*.jsonl")),
            *[root / "prepared" / f"{name}.jsonl" for name in ["teacher", "dev", "evaluation"]],
        ]
    frozen, receipt = snapshot_inputs(root, output, inputs, partial=args.partial)
    value = {}
    if args.view == "candidates":
        value = candidate_figure(frozen, output)
    if args.view in ["features", "risks", "final"]:
        points = feature_question_table(frozen, partial=args.partial)
        tails = feature_tail_table(points)
        table(output / "initial-feature-questions.csv", points)
        table(output / "initial-feature-tails.csv", tails)
        has_dev = any(v["bank"] == "development" for v in points)
        if has_dev:
            initial_metric_figure(points, output)
        value.update(feature_questions=points, feature_tails=tails, feature_scatter_rendered=has_dev)
    if args.view in ["risks", "final"]:
        values, features = risk_table(frozen, partial=args.partial)
        table(output / "risk-summary.csv", values)
        table(output / "initial-features.csv", features)
        risk_figures(values, output)
        predictions = prediction_table(values, features)
        if predictions:
            table(output / "prediction-contrasts.csv", predictions)
        value.update(risks=values, initial_features=features, prediction_contrasts=predictions)
        contrasts = method_contrasts(frozen)
        if contrasts:
            table(output / "method-contrasts.csv", contrasts)
        value.update(method_contrasts=contrasts)
    if args.view in ["capabilities", "final"]:
        capabilities, behavior = capability_tables(frozen, partial=args.partial)
        table(output / "capability-summary.csv", capabilities)
        table(output / "length-grading-summary.csv", behavior)
        value.update(capabilities=capabilities, behavior=behavior)
        value["capability_figure_rendered"] = capability_figures(capabilities, output)
        value["distance_capability_figure_rendered"] = (
            teacher_distance_figure(value["risks"], capabilities, output) if "risks" in value else False
        )
    if args.view in ["costs", "final"]:
        costs = cost_table(frozen, partial=args.partial)
        table(output / "token-costs.csv", costs)
        generated = generation_costs(frozen, partial=args.partial)
        table(output / "generation-costs.csv", generated)
        value.update(token_costs=costs, generation_costs=generated)
    save(output / "results.json", value)
    save(
        output / "provenance.json",
        dict(
            source=str(Path(__file__).resolve()),
            source_sha256=digest(Path(__file__)),
            protocol_sha256=digest(frozen / "protocol.json"),
            run=str(root),
            view=args.view,
            partial=args.partial,
            inputs=receipt,
            conditioning="One training seed; paired bootstrap over questions, not training seeds",
        ),
    )


if __name__ == "__main__":
    main()
