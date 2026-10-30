# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Verified common-state, task, and fresh-state comparisons for the twelve-condition study.

Repeated responses are averaged within question before paired resampling. Macro5
resamples questions within each dataset and gives each dataset equal weight.
Intervals describe question sampling for one training seed, not seed variability.
"""

import argparse
import csv
import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from examples.mirroropd.research.core.artifacts import file_hash, fingerprint, records
from examples.mirroropd.research.core.io import dump, finite, read, write_csv
from examples.mirroropd.research.core.statistics import paired_mean_difference
from examples.mirroropd.research.studies.fixed_transfer.evaluation import validate_records
from examples.mirroropd.research.studies.prefix_fusion.fresh import (
    ENDPOINTS,
    SCOPE,
    fresh_questions,
    load_fresh_bank,
    study_manifest,
)
from examples.mirroropd.research.studies.prefix_fusion.protocol import ARMS, PROTOCOL, protocol_identity

BOOTSTRAP = PROTOCOL["statistics"]


def paired(before, after):
    if set(before) != set(after) or not before:
        raise ValueError("Paired comparisons require the same nonempty question identities")
    keys = sorted(before)
    return paired_mean_difference(
        [before[k] for k in keys],
        [after[k] for k in keys],
        n=BOOTSTRAP["bootstrap_draws"],
        seed=BOOTSTRAP["bootstrap_seed"],
    )


def prompt_values(rows, metric, horizon, *, samples=1, window="all"):
    """Fixed-H risk, averaging repeated responses within a question before inference."""
    if horizon <= 0 or samples < 1:
        raise ValueError("Positive risk horizon and sample count required")
    grouped = defaultdict(dict)
    for row in rows:
        question = row.get("question_id", row["id"])
        sample = row.get("sample_index", 0)
        if type(sample) is not int or not 0 <= sample < samples or sample in grouped[question]:
            raise ValueError("Duplicate or invalid question/sample score slot")
        value = row["stats"][window][metric] / horizon
        if not math.isfinite(value):
            raise ValueError("Nonfinite per-question risk")
        grouped[question][sample] = value
    if not grouped or any(set(values) != set(range(samples)) for values in grouped.values()):
        raise ValueError("Missing within-question score samples")
    return {question: sum(values.values()) / samples for question, values in grouped.items()}


def accuracy_values(rows, samples=4):
    groups = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row["status"] != "ok" or row["acc"] not in [0, 1, False, True]:
            raise ValueError("Unresolved or invalid task grading result")
        source, question, sample = row["data_source"], row["prompt_id"], row["sample_index"]
        if type(sample) is not int or not 0 <= sample < samples or sample in groups[source][question]:
            raise ValueError("Duplicate or foreign task answer slot")
        groups[source][question][sample] = float(row["acc"])
    values = {}
    for source, questions in groups.items():
        if any(set(v) != set(range(samples)) for v in questions.values()):
            raise ValueError("Missing repeated task answers")
        values[source] = {question: sum(v.values()) / samples for question, v in questions.items()}
    return values


def macro5_pair(before, after):
    if len(before) != 5 or set(before) != set(after):
        raise ValueError("Macro5 requires exactly the same five datasets")
    rng = np.random.default_rng(BOOTSTRAP["bootstrap_seed"])
    draws = np.zeros(BOOTSTRAP["bootstrap_draws"], dtype=np.float64)
    means_before, means_after = [], []
    questions = 0
    for source in sorted(before):
        left, right = before[source], after[source]
        if not left or set(left) != set(right):
            raise ValueError("Macro5 requires paired questions within every dataset")
        ids = sorted(left)
        a, b = np.array([left[i] for i in ids]), np.array([right[i] for i in ids])
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite task accuracy")
        delta = b - a
        draws += delta[rng.integers(len(ids), size=(len(draws), len(ids)))].mean(axis=1) / 5
        means_before.append(float(a.mean()))
        means_after.append(float(b.mean()))
        questions += len(ids)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return dict(
        before=sum(means_before) / 5,
        after=sum(means_after) / 5,
        delta=(sum(means_after) - sum(means_before)) / 5,
        ci_low=float(lo),
        ci_high=float(hi),
        questions=questions,
        datasets=5,
        resampling="paired questions, stratified by dataset; equal dataset weights",
    )


def fixed_panel_pair(before, after, question_domains):
    """Keep the declared Math64/Code32/IF32 weights fixed in each bootstrap draw."""
    if not before or set(before) != set(after) or not set(before) <= set(question_domains):
        raise ValueError("Fixed-panel comparison requires paired known questions")
    if len({question_domains[k] for k in before}) == 1:
        return paired(before, after)
    expected = PROTOCOL["fixed_risk"]["domain_weights"]
    groups = {domain: sorted(k for k in before if question_domains[k] == domain) for domain in expected}
    if {domain: len(ids) for domain, ids in groups.items()} != expected:
        raise ValueError("Combined fixed panel must preserve its domain allocation")
    rng = np.random.default_rng(BOOTSTRAP["bootstrap_seed"])
    draws = np.zeros(BOOTSTRAP["bootstrap_draws"], dtype=np.float64)
    total = sum(expected.values())
    for domain, ids in sorted(groups.items()):
        delta = np.array([after[k] - before[k] for k in ids])
        if not np.isfinite(delta).all():
            raise ValueError("Nonfinite fixed-panel contrast")
        draws += delta[rng.integers(len(ids), size=(len(draws), len(ids)))].mean(axis=1) * expected[domain] / total
    lo, hi = np.quantile(draws, [0.025, 0.975])
    left, right = sum(before.values()) / total, sum(after.values()) / total
    return dict(
        before=left,
        after=right,
        delta=right - left,
        ci_low=float(lo),
        ci_high=float(hi),
        resampling="paired questions within domains; fixed Math64/Code32/IF32 weights",
    )


class Evidence:
    def __init__(self):
        self.files = {}

    def read(self, path):
        path = Path(path)
        digest = file_hash(path)
        value = read(path)
        if file_hash(path) != digest:
            raise ValueError(f"Evidence changed during analysis: {path}")
        self.files[str(path.resolve())] = digest
        return value

    def check_file(self, path, digest):
        path = Path(path)
        if file_hash(path) != digest:
            raise ValueError(f"Artifact checksum changed: {path}")
        self.files[str(path.resolve())] = digest

    def receipt(self, path):
        value = self.read(Path(path) / "complete.json")
        if value["fingerprint"] != fingerprint(value["inputs"]) or value["inputs"]["protocol"] != protocol_identity():
            raise ValueError(f"Receipt identity/protocol mismatch: {path}")
        return value


def load_risk(root, run, pair, outer, kind, evidence):
    dest = run / "science" / f"outer-{outer:03d}"
    stage = evidence.read(dest / "complete.json")
    if (
        stage["smoke"]
        or stage["protocol"] != protocol_identity()
        or stage["outer"] != outer
        or stage["optimizer_step"] != outer * 16
    ):
        raise ValueError("Scientific readout identity/update boundary changed")
    evidence.check_file(dest / "fixed/complete.json", stage["fixed_receipt"])
    stored = evidence.receipt(dest / kind)
    fixed = evidence.receipt(dest / "fixed")
    if stored["inputs"]["weights"] != fixed["inputs"]["weights"]:
        raise ValueError("Short and long readouts use different parameter snapshots")
    horizon = 1024 if kind == "fixed" else 7168
    target = evidence.read(root / "teacher-cache" / pair / "complete.json")
    if (
        stored["inputs"]["outer"],
        stored["inputs"]["horizon"],
        stored["inputs"]["target"],
        stored["optimizer_updates"],
    ) != (outer, horizon, target["fingerprint"], 0) or stored["inputs"].get("limit_per_source") is not None:
        raise ValueError("Risk readout has a different target, horizon, or subset")
    evidence.check_file(dest / kind / "per-question.json", stored["metrics_sha256"])
    rows = evidence.read(dest / kind / "per-question.json")
    prepared = evidence.read(root / "prepared-03/heldout-questions.json")
    expected = {r["id"]: r for r in prepared if r["primary"] and (kind == "fixed" or r["long"])}
    expected_counts = PROTOCOL["fixed_risk"]["domain_weights"] if kind == "fixed" else {"math": 32}
    if {
        domain: sum(row["domain"] == domain for row in expected.values()) for domain in expected_counts
    } != expected_counts:
        raise ValueError("Fixed-risk panel has changed domain counts")
    from examples.mirroropd.research.studies.prefix_fusion.scoring import load_bank

    banks = {}
    for source in ["S0", pair]:
        bank_rows, bank_receipt = load_bank(root / "banks" / source)
        if target["inputs"]["banks"][source] != bank_receipt["bank_sha256"]:
            raise ValueError("Fixed-risk teacher cache refers to changed generator states")
        banks[source] = {r["id"]: r for r in bank_rows}
        evidence.read(root / "banks" / source / "complete.json")
        evidence.check_file(root / "banks" / source / "bank.json", bank_receipt["bank_sha256"])
    seen = set()
    for row in rows:
        slot = row["source"], row["id"]
        if slot in seen or row["source"] not in {"S0", pair} or row["id"] not in expected:
            raise ValueError("Duplicate or foreign fixed-state question")
        original = expected[row["id"]]
        if any(row[k] != original[k] for k in ["domain", "group", "split"]):
            raise ValueError("Fixed-state question/domain identity changed")
        if row["step"] != outer * 16 or row["length"] != min(
            horizon, len(banks[row["source"]][row["id"]]["response_ids"])
        ):
            raise ValueError("Invalid fixed-state score position or length")
        seen.add(slot)
    if seen != {(source, question) for source in ["S0", pair] for question in expected}:
        raise ValueError("Incomplete common-state risk panel")
    finite(rows)
    return rows, stored


def load_accuracy(root, directory, evidence, expected_weights=None):
    stored = evidence.receipt(directory)
    inputs = stored["inputs"]
    profile = dict(samples=4, seed=PROTOCOL["accuracy"]["seed"], response_cap=31744, model_length=32768)
    if (
        inputs["smoke"]
        or inputs["panel"] != "test"
        or inputs["profile"] != profile
        or inputs["temperature"] != 0.7
        or inputs["top_p"] != 0.9
        or inputs["dtype"] != "bfloat16"
    ):
        raise ValueError("Task accuracy is not the declared full plain-model evaluation")
    if expected_weights is not None and inputs["weights"] != expected_weights:
        raise ValueError("Task accuracy and fixed-risk readout use different parameters")
    prompts = evidence.read(root / "prepared-03/task-prompts.json")
    if inputs["prompts"] != fingerprint(prompts):
        raise ValueError("Task prompts changed")
    for name, key in [
        ("records.jsonl", "records_sha256"),
        ("raw-responses.jsonl", "raw_sha256"),
        ("metrics.json", "metrics_sha256"),
    ]:
        evidence.check_file(directory / name, stored[key])
    rows = records(directory / "records.jsonl")
    raw = records(directory / "raw-responses.jsonl")
    original = {(r["prompt_id"], r["sample_index"]): r for r in raw}
    if len(original) != len(raw) or len(raw) != len(rows):
        raise ValueError("Task raw/graded response slots differ")
    for row in rows:
        before = original.get((row["prompt_id"], row["sample_index"]))
        if before is None or any(row.get(k) != value for k, value in before.items()):
            raise ValueError("Task grading changed a generated response")
    done, expected = validate_records(rows, prompts, SimpleNamespace(**profile))
    if done != expected or stored["records"] != 2520 or len(rows) != 2520 or stored["questions"] != 630:
        raise ValueError("Task evaluation has missing questions or answers")
    return accuracy_values(rows), inputs


def build(root, manifest, fresh_root, out, *, require_complete=False):
    root, fresh_root, out = map(Path, (root, fresh_root, out))
    bindings = study_manifest(manifest)
    evidence = Evidence()
    evidence.read(manifest)
    if evidence.files[str(Path(manifest).resolve())] != bindings["source_sha256"]:
        raise ValueError("Run assignments changed while starting analysis")
    prepared = evidence.read(root / "prepared-03/complete.json")
    if prepared["inputs"]["protocol"] != protocol_identity():
        raise ValueError("Prepared experiment protocol changed")
    for name in ["heldout-questions.json", "task-prompts.json"]:
        evidence.check_file(root / "prepared-03" / name, prepared["artifacts"][name])
    question_domains = {r["id"]: r["domain"] for r in evidence.read(root / "prepared-03/heldout-questions.json")}
    missing, curves, risk_comparisons, accuracies, accuracy_comparisons = [], [], [], [], []
    risks, task_values, task_inputs = {}, {}, {}
    asset_receipt = evidence.read(root / "assets/complete.json")
    for source in ["S0", "J", "P"]:
        directory = root / "baseline-accuracy" / source
        if not (directory / "complete.json").exists():
            missing.append(f"baseline-accuracy/{source}")
            continue
        declared = asset_receipt["models"][source]["files"]
        expected_weights = fingerprint(
            {
                name: value["sha256"]
                for name, value in declared.items()
                if name == "config.json" or name.endswith(".safetensors")
            }
        )
        values, inputs = load_accuracy(root, directory, evidence, expected_weights)
        task_values[("baseline", source)], task_inputs[("baseline", source)] = values, inputs
    for pair in PROTOCOL["pairs"]:
        for arm in ARMS:
            entry = bindings["pairs"][pair][arm]
            run = Path(entry["run"])
            audit_path = Path(entry["audit"])
            if audit_path.exists():
                audit = evidence.read(audit_path)
                if not audit.get("passed") or not audit.get("formal_complete") or audit.get("applied_updates") != 640:
                    missing.append(f"{pair}-{arm}/failed-or-incomplete-run-audit")
                    continue
                if (audit.get("pair"), audit.get("arm")) != (pair, arm):
                    raise ValueError("Run audit binding changed")
            else:
                missing.append(f"{pair}-{arm}/run-audit")
            for kind, outers in [
                ("fixed", PROTOCOL["fixed_risk"]["outers"]),
                ("long", PROTOCOL["long_risk"]["outers"]),
            ]:
                for outer in outers:
                    directory = run / "science" / f"outer-{outer:03d}"
                    if not (directory / "complete.json").exists() or not (directory / kind / "complete.json").exists():
                        missing.append(f"{pair}-{arm}/{outer}/{kind}")
                        continue
                    rows, stored = load_risk(root, run, pair, outer, kind, evidence)
                    domains = ["all", "math", "code", "if"] if kind == "fixed" else ["math"]
                    windows = (
                        [("all", 1024)]
                        if kind == "fixed"
                        else [("all", 7168), ("0_1024", 1024), ("1024_4096", 3072), ("4096_7168", 3072)]
                    )
                    for source, domain, (window, horizon), metric in itertools.product(
                        ["S0", pair], domains, windows, ["rkl", "js"]
                    ):
                        subset = [
                            r for r in rows if r["source"] == source and (domain == "all" or r["domain"] == domain)
                        ]
                        values = prompt_values(subset, metric, horizon, window=window)
                        key = (pair, arm, outer, kind, source, domain, window, metric)
                        risks[key] = values
                        curves.append(
                            dict(
                                pair=pair,
                                arm=arm,
                                outer=outer,
                                panel=kind,
                                state_source=source,
                                domain=domain,
                                window=window,
                                metric=metric,
                                mean=sum(values.values()) / len(values),
                                questions=len(values),
                                denominator=horizon,
                            )
                        )
                    if kind == "fixed" and outer == 40:
                        accuracy = directory / "accuracy"
                        if (accuracy / "complete.json").exists():
                            values, inputs = load_accuracy(root, accuracy, evidence, stored["inputs"]["weights"])
                            task_values[(pair, arm)], task_inputs[(pair, arm)] = values, inputs
                        else:
                            missing.append(f"{pair}-{arm}/final-accuracy")
    for key, after in risks.items():
        pair, arm, outer, *rest = key
        if outer == 0:
            continue
        before = risks.get((pair, arm, 0, *rest))
        if before:
            risk_comparisons.append(
                dict(
                    pair=pair,
                    arm=arm,
                    outer=outer,
                    panel=rest[0],
                    state_source=rest[1],
                    domain=rest[2],
                    window=rest[3],
                    metric=rest[4],
                    reference="own-outer0",
                    questions=len(after),
                    **fixed_panel_pair(before, after, question_domains),
                )
            )
    # At the final outer, compare arms only within the exact same fixed state panel.
    for pair in PROTOCOL["pairs"]:
        for first, second in itertools.combinations(ARMS, 2):
            for key, before in risks.items():
                if key[:3] != (pair, first, 40):
                    continue
                after = risks.get((pair, second, 40, *key[3:]))
                if after:
                    risk_comparisons.append(
                        dict(
                            pair=pair,
                            arm=second,
                            outer=40,
                            panel=key[3],
                            state_source=key[4],
                            domain=key[5],
                            window=key[6],
                            metric=key[7],
                            reference=first,
                            questions=len(after),
                            **fixed_panel_pair(before, after, question_domains),
                        )
                    )
    for (pair, arm), values in task_values.items():
        for source, prompts in values.items():
            accuracies.append(
                dict(
                    pair=pair,
                    arm=arm,
                    dataset=source,
                    questions=len(prompts),
                    accuracy=sum(prompts.values()) / len(prompts),
                )
            )
        if len(values) != 5:
            raise ValueError("Accuracy has an incomplete task set")
        accuracies.append(
            dict(
                pair=pair,
                arm=arm,
                dataset="macro5",
                questions=sum(map(len, values.values())),
                accuracy=sum(sum(v.values()) / len(v) for v in values.values()) / 5,
            )
        )
        if pair == "baseline":
            continue
        for reference in ["S0", pair]:
            baseline = task_values.get(("baseline", reference))
            if baseline is None:
                continue
            for field in [
                "prompts",
                "legal",
                "grader",
                "grader_implementation",
                "profile",
                "temperature",
                "top_p",
                "dtype",
            ]:
                if task_inputs[(pair, arm)][field] != task_inputs[("baseline", reference)][field]:
                    raise ValueError(f"Accuracy comparison has unmatched {field}")
            for source in baseline:
                accuracy_comparisons.append(
                    dict(
                        pair=pair,
                        arm=arm,
                        reference=reference,
                        dataset=source,
                        questions=len(values[source]),
                        **paired(baseline[source], values[source]),
                    )
                )
            accuracy_comparisons.append(
                dict(pair=pair, arm=arm, reference=reference, dataset="macro5", **macro5_pair(baseline, values))
            )
    for pair in PROTOCOL["pairs"]:
        for first, second in itertools.combinations(ARMS, 2):
            before, after = task_values.get((pair, first)), task_values.get((pair, second))
            if before is None or after is None:
                continue
            for field in [
                "prompts",
                "legal",
                "grader",
                "grader_implementation",
                "profile",
                "temperature",
                "top_p",
                "dtype",
            ]:
                if task_inputs[(pair, first)][field] != task_inputs[(pair, second)][field]:
                    raise ValueError(f"Between-arm accuracy comparison has unmatched {field}")
            for source in before:
                accuracy_comparisons.append(
                    dict(
                        pair=pair,
                        arm=second,
                        reference=first,
                        dataset=source,
                        questions=len(after[source]),
                        **paired(before[source], after[source]),
                    )
                )
            accuracy_comparisons.append(
                dict(pair=pair, arm=second, reference=first, dataset="macro5", **macro5_pair(before, after))
            )
    matrix, decompositions = read_fresh_matrix(root, fresh_root, evidence, missing, bindings)
    summary = dict(
        protocol=protocol_identity(),
        complete=not missing,
        missing=missing,
        conditions=12,
        scope=BOOTSTRAP["scope"],
        statistics=BOOTSTRAP,
        numpy_version=np.__version__,
        interval_scope="Pointwise exploratory intervals; no multiplicity correction or training-seed uncertainty",
        fresh_metric_scope=SCOPE,
        run_bindings=bindings,
        counts=dict(
            risk_points=len(curves),
            accuracy_rows=len(accuracies),
            fresh_cells=len({(row["pair"], row["parameter"], row["state"]) for row in matrix}),
        ),
    )
    finite(summary)
    if require_complete and missing:
        raise ValueError(f"Scientific results incomplete: {len(missing)} missing artifacts")
    if file_hash(manifest) != bindings["source_sha256"]:
        raise ValueError("Run assignments changed during analysis; take a new snapshot")
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "run-bindings.json", bindings)
    for name, rows in [
        ("risk-curves", curves),
        ("paired-risk", risk_comparisons),
        ("accuracy", accuracies),
        ("paired-accuracy", accuracy_comparisons),
        ("fresh-matrix", matrix),
        ("fresh-decomposition", decompositions),
    ]:
        if rows:
            finite(rows)
            write_csv(out / f"{name}.csv", rows)
    summary["evidence_sha256"] = evidence.files
    summary["analysis_sha256"] = file_hash(__file__)
    summary["artifacts_sha256"] = {path.name: file_hash(path) for path in out.glob("*.csv")}
    source = out / "source"
    source.mkdir()
    for name in ["analysis.py", "fresh.py"]:
        (source / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    dump(out / "summary.json", summary)
    report = [
        "# 六种前缀的共同评估结果",
        "",
        "状态：" + ("完整" if not missing else "部分结果；缺失条件不补零、不做最终排序。"),
        "",
        "区间为 10000 次按题配对 bootstrap，seed2026102712；仅有一个训练 seed。",
        "",
        "Fresh 矩阵为固定访问状态上的原生完整词表条件风险，不宣称等于采样轨迹 KL。",
        "",
        f"已导出 {len(curves)} 个固定风险读数、{len(accuracies)} 个准确率汇总、{len(matrix)} 个 fresh 指标单元。",
    ]
    report += [
        "",
        (
            "每题先平均重复回答，再按题重采样；macro5 对五个数据集等权。"
            "固定 H 风险不会把提前结束后的未访问位置当成新的状态，fresh 长度与 EOS/cap 单独记录。"
        ),
    ]
    if accuracies:
        report += ["", "| 配对 | 参数 | 纯模型 macro5 avg@4 |", "| --- | --- | ---: |"]
        report += [
            f"| {row['pair']} | {row['arm']} | {100 * row['accuracy']:.2f}% |"
            for row in accuracies
            if row["dataset"] == "macro5"
        ]
    if curves:
        report += [
            "",
            "| 配对 | 分支 | 最近 outer | S0 固定状态 RKL/H | Teacher 固定状态 RKL/H |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for pair in PROTOCOL["pairs"]:
            for arm in ARMS:
                selected = [
                    row
                    for row in curves
                    if row["pair"] == pair
                    and row["arm"] == arm
                    and row["panel"] == "fixed"
                    and row["domain"] == "all"
                    and row["metric"] == "rkl"
                ]
                if selected:
                    outer = max(row["outer"] for row in selected)
                    values = {row["state_source"]: row["mean"] for row in selected if row["outer"] == outer}
                    report.append(f"| {pair} | {arm} | {outer} | {values['S0']:.6g} | {values[pair]:.6g} |")
    if missing:
        report += ["", "未完成项见 summary.json 的 missing 字段。不同 outer 的部分结果不能直接作最终策略排名。"]
    (out / "README.md").write_text("\n".join(report) + "\n")
    return summary


def read_fresh_matrix(root, fresh_root, evidence, missing, bindings):
    matrix, contrasts, values = [], [], {}
    expected_ids = {r["id"] for r in fresh_questions(root)}
    bank_cache = {}
    for pair in PROTOCOL["pairs"]:
        parameter_identities = {}
        state_identities = {}
        for parameter, source in itertools.product(ENDPOINTS, repeat=2):
            path = fresh_root / "scores" / pair / parameter / source
            if not (path / "complete.json").exists():
                missing.append(f"fresh/{pair}/{parameter}/{source}")
                continue
            stored = evidence.receipt(path)
            inputs = stored["inputs"]
            if (
                inputs["smoke"]
                or inputs["horizon"] != 7168
                or (inputs["pair"], inputs["parameter"], inputs["state"]) != (pair, parameter, source)
            ):
                raise ValueError("Fresh score cell has wrong endpoint identity/profile")
            if stored["optimizer_updates"] != 0:
                raise ValueError("Fresh scorer performed optimizer updates")
            if stored["rows"] != 128 or stored["questions"] != 64:
                raise ValueError("Fresh completion receipt has incomplete question counts")
            model = inputs["model"]
            if model["endpoint"] != parameter or model["pair"] != ("shared" if parameter == "initial" else pair):
                raise ValueError("Fresh parameter identity changed")
            if parameter_identities.setdefault(parameter, model["identity"]) != model["identity"]:
                raise ValueError("A fresh matrix row uses different parameter identities")
            bank_path = fresh_root / "banks" / pair / source
            if (pair, source) not in bank_cache:
                bank_cache[(pair, source)] = load_fresh_bank(root, bank_path)
            bank, bank_receipt = bank_cache[(pair, source)]
            if (
                bank_receipt["inputs"]["model"]["endpoint"] != source
                or bank_receipt["inputs"]["model"]["pair"] != ("shared" if source == "initial" else pair)
                or inputs["bank"] != bank_receipt["bank_sha256"]
            ):
                raise ValueError("Fresh state column uses another frozen generator")
            for label, point in [(parameter, model), (source, bank_receipt["inputs"]["model"])]:
                if label != "initial":
                    entry = bindings["pairs"][pair][label]
                    if Path(point["run"]).resolve() != Path(entry["run"]).resolve() or point[
                        "run_audit_sha256"
                    ] != file_hash(entry["audit"]):
                        raise ValueError("Fresh matrix uses an unbound trained run")
            if state_identities.setdefault(source, inputs["bank"]) != inputs["bank"]:
                raise ValueError("A fresh matrix column uses different visited states")
            evidence.check_file(bank_path / "bank.json", inputs["bank"])
            evidence.read(bank_path / "complete.json")
            evidence.check_file(path / "per-question.json", stored["metrics_sha256"])
            scored = evidence.read(path / "per-question.json")
            expected = {r["id"]: r for r in bank}
            if len(scored) != 128 or len({r["id"] for r in scored}) != 128:
                raise ValueError("Fresh matrix cell is incomplete")
            for row in scored:
                original = expected[row["id"]]
                if any(row[k] != original[k] for k in ["question_id", "sample_index", "group", "domain", "source"]):
                    raise ValueError("Fresh scored question identity changed")
                if row["length"] != len(original["response_ids"]):
                    raise ValueError("Fresh scored history length changed")
            for metric in ["rkl", "js"]:
                prompt = prompt_values(scored, metric, 7168, samples=2)
                if set(prompt) != expected_ids:
                    raise ValueError("Fresh matrix has a different Math64 panel")
                values[(pair, parameter, source, metric)] = prompt
                matrix.append(
                    dict(
                        pair=pair,
                        parameter=parameter,
                        state=source,
                        metric=metric,
                        mean=sum(prompt.values()) / 64,
                        questions=64,
                        responses=128,
                        mean_response_length=sum(len(r["response_ids"]) for r in bank) / 128,
                        eos=sum(r["finish_reason"] == "stop" for r in bank),
                        cap=sum(r["finish_reason"] == "length" for r in bank),
                    )
                )
        for arm, metric in itertools.product(ARMS, ["rkl", "js"]):
            start = values.get((pair, "initial", "initial", metric))
            fixed = values.get((pair, arm, "initial", metric))
            end = values.get((pair, arm, arm, metric))
            if start is None or fixed is None or end is None:
                continue
            for label, before, after in [
                ("parameter_change_on_initial_states", start, fixed),
                ("state_change_with_final_parameters", fixed, end),
                ("total_diagonal_change", start, end),
            ]:
                contrasts.append(
                    dict(pair=pair, arm=arm, metric=metric, contrast=label, questions=64, **paired(before, after))
                )
    return matrix, contrasts


def render_baselines(directory, style_dir):
    """Plot observed three-model baseline means only.

    Ledger: x groups are five tasks plus their equal-weight aggregate; each
    bar is one model's question-mean avg@4 in percent, with equal group widths
    and a shared 0--100 axis. Colors identify fixed original checkpoints.
    Missing trained endpoints are never represented by zero-height bars.
    """
    directory, style_dir = Path(directory), Path(style_dir)
    summary = read(directory / "summary.json")
    data = directory / "accuracy.csv"
    if file_hash(data) != summary["artifacts_sha256"]["accuracy.csv"]:
        raise ValueError("Baseline figure source CSV changed")
    with data.open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["pair"] == "baseline"]
    datasets = ["MATH-500", "AMC23", "AIME24", "AIME25", "AIME26", "macro5"]
    models = [("S0", "R1 initial"), ("J", "JustRL-1.5B"), ("P", "Polaris-7B")]
    values = {(row["arm"], row["dataset"]): float(row["accuracy"]) for row in rows}
    if len(values) != len(rows) or set(values) != {(model, dataset) for model, _ in models for dataset in datasets}:
        raise ValueError("Baseline figure requires all three complete evaluations")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
        raise ValueError("Invalid observed baseline accuracy")
    sys.path.insert(0, str(style_dir))
    import fvstyle as fv
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fv.apply_rc("paper", fontsize=10)
    colors = [fv.NEUTRALS["baseline"], *fv.series_colors("paper", 2)]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(datasets))
    for index, (model, label) in enumerate(models):
        ax.bar(
            x + (index - 1) * 0.24,
            [100 * values[(model, dataset)] for dataset in datasets],
            width=0.23,
            color=colors[index],
            label=label,
        )
    ax.set_xticks(x, [*datasets[:-1], "Macro5"])
    ax.set_xlabel("Task / aggregate")
    ax.set_ylabel("Accuracy (avg@4, %)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.10), ncol=3, frameon=False)
    fig.text(
        0.5,
        0.03,
        "630 shared questions, 4 answers each; Macro5 equally weights the five tasks.",
        ha="center",
        color=fv.NEUTRALS["baseline"],
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.21, 1, 1))
    fig.canvas.draw()
    issues = fv.audit(fig, print_slot=fig.get_figwidth())
    figure_dir = directory / "figures"
    figure_dir.mkdir(exist_ok=False)
    lines = list(fv.format_audit_lines(issues))
    (figure_dir / "baseline-accuracy.audit.txt").write_text(
        "\n".join(lines) + "\n" if lines else "PASS: fv.audit found no issues.\n"
    )
    if any(fv.problem_severity(issue) == "error" for issue in issues):
        raise ValueError("Baseline figure audit failed")
    write_csv(figure_dir / "baseline-data.csv", rows)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(figure_dir / f"baseline-accuracy.{suffix}", dpi=180, facecolor=fv.NEUTRALS["paper"])
    plt.close(fig)
    dump(
        figure_dir / "receipt.json",
        dict(
            source_csv_sha256=file_hash(data),
            analysis_sha256=file_hash(__file__),
            style_sha256=file_hash(style_dir / "fvstyle.py"),
            scope="Observed original-model baselines; no trained-condition result",
            outputs={path.name: file_hash(path) for path in figure_dir.iterdir() if path.is_file()},
        ),
    )
    return figure_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fresh-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--style-dir", type=Path, help="Optional figure-viz assets/mpl path for actual baseline plot")
    args = parser.parse_args()
    result = build(
        args.root,
        args.manifest,
        args.fresh_root or args.root / "fresh",
        args.out,
        require_complete=args.require_complete,
    )
    if args.style_dir is not None:
        render_baselines(args.out, args.style_dir)
    print(result["counts"], "complete", result["complete"])


if __name__ == "__main__":
    main()
