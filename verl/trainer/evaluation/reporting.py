# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Source-aware RKL aggregation with explicit fixed-bank baselines."""

from collections import defaultdict

from .metrics import readout_metrics, summarize_rkl


def evaluation_metrics(rows, *, baselines, previous_history=(), aliases=None):
    source_baseline = baselines.get("source_heldout")
    spaces = {r.get("measure_id") for r in rows}
    if len(spaces) > 1:
        raise ValueError("Cannot aggregate different RKL distribution identities")
    for previous in (baselines or {}).values():
        if spaces and previous and {r.get("measure_id") for r in previous} != spaces:
            raise ValueError("Cannot compare a fixed bank across different RKL distributions")
    metrics = readout_metrics(rows, baseline=source_baseline, previous_history=previous_history)
    groups = defaultdict(list)
    for row in rows:
        if row["bank"].startswith("public_"):
            groups[row["bank"]].append(row)
    for bank, selected in groups.items():
        panels = {"all": selected}
        for row in selected:
            panels.setdefault(row["domain"], []).append(row)
        for domain, panel in panels.items():
            prefix = f"eval/rkl/{bank}/{domain}"
            group = "domain" if domain == "all" else None
            summary = summarize_rkl(panel, equal_groups=group)
            metrics.update({f"{prefix}/{k}": v for k, v in summary.items() if k != "mean"})
            if "mean" in summary:
                metrics[f"{prefix}/question_mean"] = summary["mean"]
            for diagnostic in ("student_excluded_mass", "teacher_excluded_mass"):
                if panel and all(r.get(diagnostic + "_sum") is not None for r in panel):
                    mass_rows = [dict(r, rkl_sum=r[diagnostic + "_sum"]) for r in panel]
                    summary_mass = summarize_rkl(mass_rows, equal_groups=group)
                    if "mean" in summary_mass:
                        metrics[f"{prefix}/{diagnostic}"] = summary_mass["mean"]
            baseline = baselines.get(bank)
            if baseline is not None and bank != "public_current":
                original = [r for r in baseline if domain == "all" or r["domain"] == domain]
                if {(r["id"], r["tokens"]) for r in original} != {(r["id"], r["tokens"]) for r in panel}:
                    raise ValueError("Fixed public bank identities or masks changed")
                before = summarize_rkl(original, equal_groups=group)
                if "mean" in before and "mean" in summary:
                    metrics[f"{prefix}/absolute_reduction"] = before["mean"] - summary["mean"]
                    if before["mean"] > 0:
                        metrics[f"{prefix}/reduction_percent"] = 100 * (1 - summary["mean"] / before["mean"])
    for bank in ("source_heldout", "history"):
        panel = [r for r in rows if r["bank"] == bank]
        for diagnostic in ("student_excluded_mass", "teacher_excluded_mass"):
            if panel and all(r.get(diagnostic + "_sum") is not None for r in panel):
                summary = summarize_rkl(
                    [dict(r, rkl_sum=r[diagnostic + "_sum"]) for r in panel],
                    equal_groups="history_batch" if bank == "history" else None,
                )
                if "mean" in summary:
                    metrics[f"eval/rkl/{bank}/{diagnostic}"] = summary["mean"]
    for alias, original in (aliases or {}).items():
        prefix = f"eval/rkl/{original}/"
        for key, value in list(metrics.items()):
            if key.startswith(prefix) and not key.endswith(("reduction_percent", "absolute_reduction")):
                metrics[key.replace(prefix, f"eval/rkl/{alias}/", 1)] = value
    return metrics
