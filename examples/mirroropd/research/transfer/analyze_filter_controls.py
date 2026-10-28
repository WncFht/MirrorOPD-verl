# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Additional functional, synthetic, support and matched-acquisition analyses."""

import argparse
import math
import statistics as st
from pathlib import Path

from ..common.io import dump, read
from ..common.statistics import mean_ci, ratio_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    a = ap.parse_args()
    root = a.data
    confounds = []
    for teacher in ["justrl", "polaris"]:
        proxy = {r["id"]: r for r in read(root / f"transfer-proxy-{teacher}.json")["records"]}
        baseline = root / (teacher + "-student-reverse") / "fixed-000.json"
        if not baseline.exists():
            continue
        rows = [r for r in read(baseline) if r["view"] == "student" and r["role"] == "train"]
        for metric in ["js", "hp", "q_outside_initial_top16"]:
            confounds.append(
                dict(
                    teacher=teacher,
                    metric=metric,
                    diagnostic_cosine_correlation=st.correlation(
                        [proxy[r["id"]]["diagnostic_cosine"] for r in rows],
                        [r["stats"]["all"][metric] / 1024 for r in rows],
                    ),
                )
            )
    dump(root / "proxy-confounds.json", confounds)
    synthetic = []
    for kind in ["shared", "random"]:
        for direction in ["reverse", "forward"]:
            dest = root / f"synthetic-{kind}-{direction}"
            for file in sorted(dest.glob("eval-*.json")):
                rows = read(file)
                for split, lo, hi in [("train", 0, 16), ("test", 16, 80)]:
                    synthetic.append(
                        dict(
                            kind=kind,
                            direction=direction,
                            step=int(file.stem.split("-")[-1]),
                            split=split,
                            **{k: st.mean(v[lo:hi]) for k, v in rows.items()},
                        )
                    )
    dump(root / "synthetic-summary.json", synthetic)
    functions = []
    baseline = root / "if-eval/baseline-student.json"
    if baseline.exists():
        base = read(baseline)
        if len(base) == 64 and all("grade" in r for r in base):
            initial = {r["id"]: r for r in base}
            for file in sorted((root / "if-eval").glob("*.json")):
                rows = read(file)
                if len(rows) != 64 or not all("grade" in r for r in rows):
                    continue
                differences = [int(r["grade"]["acc"]) - int(initial[r["id"]]["grade"]["acc"]) for r in rows]
                functions.append(
                    dict(
                        name=file.stem,
                        correct=sum(r["grade"]["acc"] for r in rows),
                        strict_and_natural_stop=sum(r["grade"]["acc"] and r["finish_reason"] == "stop" for r in rows),
                        loose_correct=sum(r["grade"]["prompt_loose_acc"] for r in rows),
                        strict_constraint_accuracy=sum(r["grade"]["if_inst_strict_correct"] for r in rows)
                        / sum(r["grade"]["if_inst_total"] for r in rows),
                        lost_correct=sum(initial[r["id"]]["grade"]["acc"] and not r["grade"]["acc"] for r in rows),
                        new_correct=sum(not initial[r["id"]]["grade"]["acc"] and r["grade"]["acc"] for r in rows),
                        delta=st.mean(differences),
                        paired_prompt_ci=mean_ci(differences),
                        mean_length=st.mean(len(r["response_ids"]) for r in rows),
                        length_stops=sum(r["finish_reason"] == "length" for r in rows),
                    )
                )
    dump(root / "if-summary.json", functions)
    # Interpolate only between adjacent checkpoints that bracket the target.
    # Select earliest upward crossing; no extrapolation or performance-based arm selection.
    risks = read(root / "risk-summary.json")
    matched = []
    for teacher in ["justrl", "polaris"]:
        for target in [0.05, 0.10, 0.20, 0.30, 0.50, 0.80]:
            for arm in read(root / "protocol.json")["arms"]:
                if arm["teacher"] != teacher:
                    continue
                panel = "teacher-train-math" if arm["source"] == "teacher" else "student-train-math"
                curve = sorted(
                    [r for r in risks if r["name"] == arm["name"] and r["panel"] == panel], key=lambda r: r["step"]
                )
                for left, right in zip(curve, curve[1:], strict=False):
                    if left["js_drop"] <= target <= right["js_drop"] and right["js_drop"] > left["js_drop"]:
                        weight = (target - left["js_drop"]) / (right["js_drop"] - left["js_drop"])
                        test = {
                            r["step"]: r
                            for r in risks
                            if r["name"] == arm["name"] and r["panel"] == "student-test-math"
                        }
                        lower, upper = test[left["step"]], test[right["step"]]
                        matched.append(
                            arm
                            | dict(
                                target_train_js_drop=target,
                                reference_panel=panel,
                                lower_step=left["step"],
                                upper_step=right["step"],
                                interpolation_weight=weight,
                                test_js_drop=lower["js_drop"] + weight * (upper["js_drop"] - lower["js_drop"]),
                                test_initial_policy_fkl=lower["retain_fkl"]
                                + weight * (upper["retain_fkl"] - lower["retain_fkl"]),
                                scope=(
                                    "Descriptive interpolation; compare losses within same source only. "
                                    "Online uses frozen S0 train reference, not current training states."
                                ),
                            )
                        )
                        break
    dump(root / "matched-acquisition.json", matched)
    support = []
    for row in risks:
        if row["step"] in [0, 32, 64, 96, 128]:
            support.append(
                {
                    k: row[k]
                    for k in [
                        "name",
                        "teacher",
                        "source",
                        "direction",
                        "step",
                        "panel",
                        "js_drop",
                        "hp",
                        "hq",
                        "overlap16",
                        "common_p16",
                        "common_q16",
                        "q_outside_initial_top16",
                        "p_outside_initial_top16",
                        "p0_outside_initial_top16",
                        "teacher_only_initial_p",
                        "teacher_only_initial_p0",
                        "teacher_only_initial_q",
                    ]
                }
            )
    dump(root / "support-summary.json", support)
    matched_views = []
    for arm in read(root / "protocol.json")["arms"]:
        baseline = root / arm["name"] / "fixed-000.json"
        if not baseline.exists():
            continue
        before = read(baseline)
        ids = {r["id"] for r in before if r["view"] == "teacher" and r["role"] == "test"}
        assert len(ids) == 64
        initial = {(r["view"], r["id"]): r for r in before if r["id"] in ids}
        for file in sorted(baseline.parent.glob("fixed-*.json")):
            for view in ["student", "teacher"]:
                rows = [r for r in read(file) if r["id"] in ids and r["view"] == view]
                assert len(rows) == 64
                b = st.mean(initial[view, r["id"]]["stats"]["all"]["js"] / 1024 for r in rows)
                v = st.mean(r["stats"]["all"]["js"] / 1024 for r in rows)
                entry = arm | dict(
                    view=view, step=int(file.stem.split("-")[-1]), prompts=64, js=v, js_before=b, js_drop=1 - v / b
                )
                if entry["step"] == 128:
                    ci = ratio_ci(
                        [initial[view, r["id"]]["stats"]["all"]["js"] for r in rows],
                        [r["stats"]["all"]["js"] for r in rows],
                    )
                    entry["paired_prompt_js_drop_ci"] = [1 - ci[1], 1 - ci[0]]
                matched_views.append(entry)
    dump(root / "matched-prefix-views.json", matched_views)
    pairs = []
    for teacher in ["justrl", "polaris"]:
        for left, right in [
            ("student-reverse", "student-forward"),
            ("teacher-reverse", "teacher-forward"),
            ("online-reverse", "online-forward"),
            ("online-reverse", "teacher-sft"),
        ]:
            left, right = teacher + "-" + left, teacher + "-" + right
            for domain in ["math", "if"]:
                paths = (
                    [root / left / "task.json", root / right / "task.json"]
                    if domain == "math"
                    else [root / "if-eval" / (left + ".json"), root / "if-eval" / (right + ".json")]
                )
                if not all(p.exists() for p in paths):
                    continue
                lhs, rhs = [read(p) for p in paths]
                if not all(len(rs) == 64 and all("grade" in r for r in rs) for rs in [lhs, rhs]):
                    continue
                ri = {r["id"]: r for r in rhs}
                delta = [int(r["grade"]["acc"]) - int(ri[r["id"]]["grade"]["acc"]) for r in lhs]
                pairs.append(
                    dict(
                        domain=domain,
                        left=left,
                        right=right,
                        accuracy_difference=st.mean(delta),
                        paired_prompt_ci=mean_ci(delta),
                        left_only_correct=sum(v == 1 for v in delta),
                        right_only_correct=sum(v == -1 for v in delta),
                    )
                )
    dump(root / "paired-functional-controls.json", pairs)
    calibration = []
    initial_risk = next(
        r["js_before"] for r in risks if r["name"] == "polaris-student-reverse" and r["panel"] == "student-test-math"
    )
    for file in sorted((root / "calibration").glob("*.json")):
        result = read(file)
        if not result["complete"]:
            continue
        temperature = str(result["selected_temperature"])
        before = result["risks"]["test"]["1.0"]["js"]
        after = result["risks"]["test"][temperature]["js"]
        rs = [r for r in result["rows"] if r["split"] == "test"]
        differences = [(r["scores"][temperature]["js"] - r["scores"]["1.0"]["js"]) / 1024 for r in rs]
        if result["arm"] == "initial":
            reference = initial_risk
        else:
            reference = next(
                r["js"]
                for r in risks
                if r["name"] == result["arm"] and r["step"] == 128 and r["panel"] == "student-test-math"
            )
        assert abs(before - reference) < 1e-6, (result["arm"], before, reference)
        q_reference = next(
            r["js_before"] if result["arm"] == "initial" else r["js"]
            for r in risks
            if r["name"] == ("polaris-student-reverse" if result["arm"] == "initial" else result["arm"])
            and r["step"] == 128
            and r["panel"] == "teacher-test-math"
        )
        assert abs(result["risks"]["teacher_test"]["1.0"]["js"] - q_reference) < 1e-6
        calibration.append(
            dict(
                arm=result["arm"],
                temperature=result["selected_temperature"],
                grid_boundary=result["selection_at_grid_boundary"],
                before=before,
                after=after,
                uncalibrated_drop=1 - before / initial_risk,
                calibrated_drop=1 - after / initial_risk,
                paired_test_js_difference_ci=mean_ci(differences),
                temperature_one_reference_error=abs(before - reference),
                teacher_prefix_js_before=result["risks"]["teacher_test"]["1.0"]["js"],
                teacher_prefix_js_after=result["risks"]["teacher_test"][temperature]["js"],
                teacher_sequence_fkl_before=1024 * result["risks"]["teacher_test"]["1.0"]["fkl"],
                teacher_sequence_fkl_after=1024 * result["risks"]["teacher_test"][temperature]["fkl"],
            )
        )
    dump(root / "calibration-summary.json", calibration)
    retention = []
    matrices = []
    for arm in read(root / "protocol.json")["arms"]:
        if arm["source"] == "online":
            continue
        dest = root / arm["name"]
        if not (dest / "fixed-128.json").exists():
            continue
        view = "student" if arm["source"] == "student" else "teacher"
        final = read(dest / "fixed-128.json")
        for batch in range(1, 8):
            before = read(dest / f"active-{batch:02d}-before.json")
            fitted = read(dest / f"active-{batch:02d}-after.json")
            ids = {r["id"] for r in fitted}
            later = [r for r in final if r["view"] == view and r["id"] in ids]
            assert len(later) == 16
            initial_risk = st.mean(r["stats"]["all"]["js"] / 1024 for r in before)
            fitted_risk = st.mean(r["stats"]["all"]["js"] / 1024 for r in fitted)
            final_risk = st.mean(r["stats"]["all"]["js"] / 1024 for r in later)
            objectives = {}
            for metric in ["rkl", "fkl_shared", "reference_nll"]:
                fit = st.mean(r["stats"]["all"][metric] / 1024 for r in fitted)
                end = st.mean(r["stats"]["all"][metric] / 1024 for r in later)
                objectives[metric] = dict(post_batch=fit, final=end, change=end - fit, ratio=end / fit)
            retention.append(
                arm
                | dict(
                    batch=batch,
                    pre_batch_js=initial_risk,
                    post_batch_js=fitted_risk,
                    final_js=final_risk,
                    change_after_leaving_batch=final_risk - fitted_risk,
                    ratio_after_leaving_batch=final_risk / fitted_risk,
                    objective_metrics=objectives,
                    scope="Same actual training prefixes, later disjoint batches only; not prior-task forgetting.",
                )
            )
        groups = list("ABCDEFGH") + [f"M{i:02d}" for i in range(5, 13)]

        def values(file, groups=groups, view=view, arm_name=arm["name"]):
            rows = read(file)
            out = []
            for g in groups:
                selected = [
                    r for r in rows if r["group"] == g and r["view"] == (view if g in "ABCDEFGH" else "student")
                ]
                assert len(selected) == 16, (arm_name, g, len(selected))
                out.append(st.mean(r["stats"]["all"]["js"] / 1024 for r in selected))
            return out

        baseline = values(dest / "fixed-000.json")
        rows = []
        for step in [0, 32, 64, 96, 128]:
            risk = values(dest / f"fixed-{step:03d}.json")
            rows.append(dict(step=step, risk=risk, ratio=[v / b for v, b in zip(risk, baseline, strict=True)]))
        matrices.append(arm | dict(groups=groups, rows=rows, training_reference=view, test_reference="student"))
    dump(root / "old-batch-retention.json", retention)
    dump(root / "batch-js-matrices.json", matrices)
    positions = []
    support_bounds = []
    for arm in read(root / "protocol.json")["arms"]:
        dest = root / arm["name"]
        if not (dest / "fixed-128.json").exists():
            continue
        before = read(dest / "fixed-000.json")
        after = read(dest / "fixed-128.json")
        for view, domain in [("student", "math"), ("student", "code"), ("student", "if"), ("teacher", "math")]:
            initial = [r for r in before if r["view"] == view and r["domain"] == domain and r["role"] == "test"]
            final = [r for r in after if r["view"] == view and r["domain"] == domain and r["role"] == "test"]
            total_js = sum(r["stats"]["all"]["js"] for r in final)
            outside_bound = (
                math.log(2)
                / 2
                * sum(
                    2 * r["length"] - r["stats"]["all"]["common_p16"] - r["stats"]["all"]["common_q16"] for r in final
                )
            )
            support_bounds.append(
                arm
                | dict(
                    view=view,
                    domain=domain,
                    step=128,
                    outside_js_upper_bound=outside_bound / (len(final) * 1024),
                    inside_intersection_js_fraction_lower_bound=max(0.0, 1 - outside_bound / total_js),
                    definition="Coordinate JS <= (p+q)*log(2)/2; current student/teacher top-16 intersection.",
                )
            )
            for band in ["early", "middle", "late"]:
                result = arm | dict(view=view, domain=domain, band=band, prompts=len(final))
                for metric in ["js", "rkl", "fkl_shared"]:
                    b = sum(r["stats"][band][metric] for r in initial)
                    value = sum(r["stats"][band][metric] for r in final)
                    result[metric + "_drop"] = 1 - value / b if b > 0 else None
                positions.append(result)
    dump(root / "position-summary.json", positions)
    dump(root / "support-error-bounds.json", support_bounds)
    sequence = []
    for arm in read(root / "protocol.json")["arms"]:
        dest = root / arm["name"]
        if not (dest / "fresh-128.json").exists():
            continue
        current = read(dest / "fresh-128.json")
        if len(current) != 64:
            continue
        initial = read(root / f"baseline-fresh-{arm['teacher']}.json")
        prompt_ids = {r["id"] for r in current}
        assert len(prompt_ids) == 32
        teacher_before = [r for r in read(dest / "fixed-000.json") if r["view"] == "teacher" and r["id"] in prompt_ids]
        teacher_after = [r for r in read(dest / "fixed-128.json") if r["view"] == "teacher" and r["id"] in prompt_ids]
        assert len(teacher_before) == len(teacher_after) == 32
        pb = {i: st.mean(r["stats"]["all"]["rkl"] for r in initial if r["id"] == i) for i in prompt_ids}
        pa = {i: st.mean(r["stats"]["all"]["rkl"] for r in current if r["id"] == i) for i in prompt_ids}
        qb = {r["id"]: r["stats"]["all"]["fkl_shared"] for r in teacher_before}
        qa = {r["id"]: r["stats"]["all"]["fkl_shared"] for r in teacher_after}
        record = arm | dict(prompts=32, student_samples_per_prompt=2, teacher_samples_per_prompt=1, horizon=1024)
        for label, before, after in [("student_sequence_rkl", pb, pa), ("teacher_sequence_fkl", qb, qa)]:
            b = st.mean(before.values())
            v = st.mean(after.values())
            intervals = ratio_ci([before[i] for i in sorted(prompt_ids)], [after[i] for i in sorted(prompt_ids)])
            record[label] = dict(
                before=b, after=v, drop=1 - v / b, paired_prompt_drop_ci=[1 - intervals[1], 1 - intervals[0]]
            )
        sequence.append(record)
    dump(root / "matched-sequence-risks.json", sequence)
    overlaps = []
    for arm in read(root / "protocol.json")["arms"]:
        for step in [0, 64, 128]:
            path = (
                root / f"baseline-fresh-{arm['teacher']}.json"
                if step == 0
                else root / arm["name"] / f"fresh-{step:03d}.json"
            )
            if not path.exists():
                continue
            rows = read(path)
            if len(rows) != 64:
                continue
            tokens = sum(len(r["response_ids"]) for r in rows)
            record = arm | dict(step=step, tokens=tokens)
            for k in [8, 16, 64]:
                metrics = [f"overlap{k}", f"jaccard{k}", f"common_p{k}", f"common_q{k}"]
                metrics += [f"{prefix}{k}_{c}" for prefix in ["category", "mass_p", "mass_q"] for c in range(4)]
                record.update({metric: sum(r["stats"]["all"][metric] for r in rows) / tokens for metric in metrics})
                assert abs(sum(record[f"category{k}_{c}"] for c in range(4)) - 1) < 1e-8
            overlaps.append(record)
    dump(root / "fresh-overlap-summary.json", overlaps)
    continuations = []
    for file in sorted((root / "continuation").glob("*.json")):
        result = read(file)
        if not result["complete"]:
            continue
        continuations.append(
            dict(
                arm=result["arm"],
                selected_cases=result["selected_cases"],
                recovered_correct=result["recovered_correct"],
                natural_stops=sum(r["finish_reason"] == "stop" for r in result["rows"]),
                mean_extra_tokens=st.mean(len(r["suffix_ids"]) for r in result["rows"]),
                scope="Selected capped failures only; not a complete longer-budget benchmark.",
            )
        )
    dump(root / "continuation-summary.json", continuations)
    print(
        "Synthetic split/step rows", len(synthetic), "graded IF models", len(functions), "matched points", len(matched)
    )


if __name__ == "__main__":
    main()
