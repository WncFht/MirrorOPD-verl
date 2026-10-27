# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Per-prompt evaluation, finite-population budget curves and replayable records.

All randomness is local. Confidence intervals resample prompts, not responses;
subset variability and Monte Carlo error are reported separately.
"""

import itertools
import json
import math
import re
import zlib
from collections import defaultdict
from numbers import Real

import numpy as np

ALIASES = {"MATH-500": "math500", "AMC23": "amc23", "AIME24": "aime24", "AIME25": "aime25", "AIME26": "aime26"}


def dataset_slug(name):
    slug = ALIASES.get(str(name), re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_"))
    if not slug:
        raise ValueError(f"Dataset name has no usable metric component: {name!r}")
    return slug


def _finite(value):
    return isinstance(value, Real) and math.isfinite(float(value))


def pass_at_k(n, c, k):
    """HumanEval estimator: probability of >=1 success in a size-k subset."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("Require 0 <= correct <= samples and 1 <= k <= samples")
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _order_moments(values, k, best):
    values = np.sort(np.asarray(values, dtype=np.float64))
    if not best:
        values = values[::-1]
    n = len(values)
    weights = np.array([math.comb(i, k - 1) / math.comb(n, k) if i >= k - 1 else 0 for i in range(n)])
    mean = float(weights @ values)
    return mean, math.sqrt(max(0.0, float(weights @ values**2) - mean**2))


def _prediction_key(value):
    if value is None or (isinstance(value, str) and value == "") or (isinstance(value, Real) and not _finite(value)):
        return None
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        default=lambda obj: obj.tolist() if hasattr(obj, "tolist") else str(obj),
    )


def _majority(values, predictions, subsets):
    """Missing predictions abstain; ties are uniformly averaged, not order-broken."""
    labels = {}
    codes = np.full(len(predictions), -1, dtype=np.int64)
    for i, pred in enumerate(predictions):
        label = _prediction_key(pred)
        if label is None:
            continue
        codes[i] = labels.setdefault(label, len(labels))
    if not labels:
        return np.zeros(len(subsets))
    scores = np.array([np.mean(np.asarray(values)[codes == i]) for i in range(len(labels))])
    selected = codes[subsets]
    counts = (selected[..., None] == np.arange(len(labels))).sum(axis=1)
    maximum = counts.max(axis=1, keepdims=True)
    ties = (counts == maximum) & (maximum > 0)
    return (ties * scores).sum(axis=1) / np.maximum(ties.sum(axis=1), 1)


def evaluation_metrics(
    data_sources,
    uids,
    infos,
    expected_counts=None,
    *,
    sample_indices=None,
    prompt_ids=None,
    source_fields=None,
    resamples=256,
    seed=42,
    _include_observed=True,
):
    """Report every supported k on the same prompt population.

    Missing main scores consume the requested budget with score zero. A source
    defining only reward never inherits another source's accuracy capability.
    An entirely failed source with unknown scoring capability reports counts
    only. source_fields can declare numeric scores; known math sources declare acc.
    """
    if type(resamples) is not int or resamples < 2:
        raise ValueError("eval_metrics_resamples must be an integer >= 2")
    seed = 42 if seed is None else seed
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer) or seed < 0:
        raise ValueError("Evaluation seed must be a nonnegative integer or None")
    n_rows = len(data_sources)
    if len(uids) != n_rows or any(len(v) != n_rows for v in infos.values()):
        raise ValueError("Validation fields must have identical row counts")
    if sample_indices is not None and len(sample_indices) != n_rows:
        raise ValueError("Sample indices must have identical row counts")
    if prompt_ids is not None and len(prompt_ids) != n_rows:
        raise ValueError("Prompt identities must have identical row counts")
    groups = defaultdict(list)
    identities = {}
    for i, (source, uid) in enumerate(zip(data_sources, uids, strict=True)):
        key = (str(source), str(uid))
        index = int(sample_indices[i]) if sample_indices is not None else len(groups[key])
        groups[key].append((index, {field: vals[i] for field, vals in infos.items()}))
        identities[key] = str(prompt_ids[i]) if prompt_ids is not None else str(uid)
    expected = {}
    for (s, u), count in (expected_counts or {}).items():
        if isinstance(count, bool) or int(count) != count or count < 0:
            raise ValueError("Expected response counts must be nonnegative integers")
        key = (str(s), str(u))
        expected[key] = int(count)
        groups[key]
        identities.setdefault(key, str(u))
    by_source = defaultdict(list)
    for key, observed in groups.items():
        n = expected.get(key, len(observed))
        if n == 0 and not observed:
            continue
        indices = [i for i, _ in observed]
        if len(set(indices)) != len(indices) or any(i < 0 or i >= n for i in indices) or len(observed) > n:
            raise ValueError("More validation observations than expected sessions, or duplicate/invalid sample indices")
        padded = [{} for _ in range(n)]
        for i, row in observed:
            padded[i] = row
        by_source[key[0]].append((identities[key], padded, len(observed)))
    result, names = {}, {}
    for source, prompts in sorted(by_source.items()):
        slug = dataset_slug(source)
        if slug in names and names[slug] != source:
            raise ValueError(f"Dataset metric name collision: {source!r}, {names[slug]!r}")
        names[slug] = source
        prompts.sort(key=lambda item: item[0])
        prefix = f"eval/{slug}/"
        all_rows = [row for _, rows, _ in prompts for row in rows]
        numeric_fields = {
            field
            for row in all_rows
            for field, value in row.items()
            if isinstance(value, Real)
            and not field.startswith("__")
            and field
            not in {
                "pred",
                "uid",
                "prompt_id",
                "sample_index",
                "step",
                "response_length",
                "num_turns",
                "tool_call_counts",
            }
        }
        declared = (source_fields or {}).get(source, [])
        if isinstance(declared, str) or any(not isinstance(field, str) for field in declared):
            raise ValueError("Source metric_fields must be a collection of score names")
        numeric_fields.update(field for field in declared if field != "pred")
        if source in ALIASES:
            numeric_fields.add("acc")
        main = (
            "acc"
            if "acc" in numeric_fields
            else ("reward" if "reward" in numeric_fields else (sorted(numeric_fields)[0] if numeric_fields else None))
        )
        counts = [len(rows) for _, rows, _ in prompts]
        k_max = min(counts)
        total = sum(counts)
        observed = sum(_finite(row.get(main)) for row in all_rows)
        result.update(
            {
                prefix + "prompt_count": len(prompts),
                prefix + "quality_defined": float(main is not None),
                prefix + "response_count": total,
                prefix + "observed_response_count": observed,
                prefix + "materialized_response_count": sum(n for _, _, n in prompts),
                prefix + "failed_count": total - observed,
                prefix + "nonfinite_score_count": sum(
                    isinstance(row.get(main), Real) and not _finite(row[main]) for row in all_rows
                ),
                prefix + "samples_per_prompt_min": min(counts),
                prefix + "samples_per_prompt_max": max(counts),
            }
        )
        rng = np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(source.encode())]))
        boot = rng.integers(len(prompts), size=(resamples, len(prompts)))

        def report(name, values, ci=True, *, prefix=prefix, boot=boot):
            values = np.asarray(values, dtype=np.float64)
            result[prefix + name] = float(values.mean())
            if ci:
                lower, upper = np.quantile(values[boot].mean(axis=1), [0.025, 0.975])
                result[prefix + name + "_ci_low"] = float(lower)
                result[prefix + name + "_ci_high"] = float(upper)

        for raw_field, label in [
            ("__response_length__", "response_length"),
            ("__num_turns__", "num_turns"),
            ("__tool_calls__", "tool_calls"),
        ]:
            if raw_field not in infos:
                continue
            measured = [float(r[raw_field]) for r in all_rows if _finite(r.get(raw_field))]
            result[prefix + f"generation/{label}/observed_count"] = len(measured)
            if measured:
                for stat, value in [
                    ("mean", np.mean(measured)),
                    ("min", min(measured)),
                    ("max", max(measured)),
                    ("p50", np.quantile(measured, 0.5)),
                    ("p95", np.quantile(measured, 0.95)),
                ]:
                    result[prefix + f"generation/{label}/{stat}"] = float(value)
        if "__finish_reason__" in infos:
            reasons = [r.get("__finish_reason__") for r in all_rows]
            for reason in ("stop", "length", "aborted", "completed", "unknown"):
                count = sum((r or "unknown") == reason or (reason == "aborted" and r == "abort") for r in reasons)
                result[prefix + f"generation/finish_reason/{reason}_count"] = count
        field_names = {}
        for field in sorted(numeric_fields):
            component = "format" if field == "format_score" else dataset_slug(field)
            if component in field_names:
                raise ValueError(f"Evaluation score name collision: {field!r}, {field_names[component]!r}")
            field_names[component] = field
            values = [
                np.array([float(r[field]) if _finite(r.get(field)) else 0.0 for r in rows]) for _, rows, _ in prompts
            ]
            report(component + "/mean", [v.mean() for v in values])
            report(component + "/within_prompt_std", [v.std() for v in values], ci=False)
            observed_means = [
                np.mean([r[field] for r in rows if _finite(r.get(field))])
                for _, rows, _ in prompts
                if any(_finite(r.get(field)) for r in rows)
            ]
            if observed_means:
                result[prefix + component + "/observed_mean"] = float(np.mean(observed_means))
            result[prefix + component + "/missing_count"] = sum(not _finite(r.get(field)) for r in all_rows)
            if field == "acc" and any(np.any((v != 0) & (v != 1)) for v in values):
                raise ValueError("pass@k requires binary accuracy scores")
            for k in range(1, k_max + 1):
                report(component + f"/prefix_mean@{k}", [v[:k].mean() for v in values], ci=False)
                if field == "acc":
                    ps = [pass_at_k(len(v), int(v.sum()), k) for v in values]
                    report(f"acc/pass@{k}", ps)
                    report(f"acc/pass@{k}_subset_std", [math.sqrt(p * (1 - p)) for p in ps], ci=False)
                    all_correct = [
                        math.comb(int(v.sum()), k) / math.comb(len(v), k) if v.sum() >= k else 0 for v in values
                    ]
                    report(f"acc/all_correct@{k}", all_correct)
                    report(f"acc/all_correct@{k}_subset_std", [math.sqrt(v * (1 - v)) for v in all_correct], ci=False)
                else:
                    for label, best in (("best", True), ("worst", False)):
                        moments = [_order_moments(v, k, best) for v in values]
                        report(f"{component}/{label}@{k}", [m[0] for m in moments])
                        report(f"{component}/{label}@{k}_subset_std", [m[1] for m in moments], ci=False)
        # Voting is an accuracy operation, never a generic transformation of all scores.
        voting_defined = (
            source in ALIASES or "pred" in declared or any(_prediction_key(r.get("pred")) is not None for r in all_rows)
        )
        if "acc" in numeric_fields:
            result[prefix + "acc/voting_defined"] = float(voting_defined)
            result[prefix + "acc/prediction_missing_count"] = sum(
                _prediction_key(r.get("pred")) is None for r in all_rows
            )
        if "acc" in numeric_fields and voting_defined:
            votes = defaultdict(list)
            variances = defaultdict(list)
            mc_variances = defaultdict(list)
            conflict_prompts = 0
            for _, rows, _ in prompts:
                n = len(rows)
                values = [float(r["acc"]) if _finite(r.get("acc")) else 0.0 for r in rows]
                preds = [r.get("pred") if _finite(r.get("acc")) else None for r in rows]
                labels = defaultdict(set)
                for pred, value in zip(preds, values, strict=True):
                    key = _prediction_key(pred)
                    if key is not None:
                        labels[key].add(value)
                conflict_prompts += any(len(v) > 1 for v in labels.values())
                permutations = np.argsort(rng.random((resamples, n)), axis=1)
                for k in range(1, k_max + 1):
                    exact = n <= 8 or k in (1, n)
                    if exact:
                        subsets = np.array(list(itertools.combinations(range(n), k)))
                    else:
                        subsets = permutations[:, :k]
                    vals = _majority(values, preds, subsets)
                    votes[k].append(float(vals.mean()))
                    variances[k].append(float(vals.std()))
                    mc_variances[k].append(0.0 if exact else float(vals.var(ddof=1) / len(vals)))
            result[prefix + "acc/prediction_score_conflict_fraction"] = conflict_prompts / len(prompts)
            for k in range(1, k_max + 1):
                report(f"acc/majority@{k}", votes[k])
                report(f"acc/majority@{k}_subset_std", variances[k], ci=False)
                result[prefix + f"acc/majority@{k}_mc_se"] = math.sqrt(sum(mc_variances[k])) / len(prompts)
        # Success-only curves are identical when every score is present. When
        # failures occur, preserve the conditional view with its own population
        # counts instead of silently changing the main denominator.
        if _include_observed:
            for component, field in field_names.items():
                missing = result[prefix + component + "/missing_count"]
                observed_rows = [
                    (identity, row) for identity, rows, _ in prompts for row in rows if _finite(row.get(field))
                ]
                if not missing or not observed_rows:
                    continue
                child_source = source + "__observed"
                child_infos = {field: [row[field] for _, row in observed_rows]}
                if field == "acc":
                    child_infos["pred"] = [row.get("pred") for _, row in observed_rows]
                child = evaluation_metrics(
                    [child_source] * len(observed_rows),
                    [identity for identity, _ in observed_rows],
                    child_infos,
                    source_fields={child_source: [field] + (["pred"] if field == "acc" and voting_defined else [])},
                    resamples=resamples,
                    seed=seed,
                    _include_observed=False,
                )
                child_prefix = f"eval/{dataset_slug(child_source)}/"
                for name, value in child.items():
                    suffix = name[len(child_prefix) :]
                    if suffix.startswith(component + "/"):
                        result[prefix + component + "_observed/" + suffix[len(component) + 1 :]] = value
                    elif suffix in (
                        "prompt_count",
                        "response_count",
                        "samples_per_prompt_min",
                        "samples_per_prompt_max",
                    ):
                        result[prefix + component + "_observed/" + suffix] = value
    return result


def evaluation_records(
    data_sources,
    uids,
    infos,
    expected_counts,
    *,
    sample_indices,
    prompt_metadata,
    inputs,
    outputs,
    scores,
    ground_truths,
    response_lengths,
    turns,
):
    """Final-session rows, including explicit placeholders for missing generations."""
    accuracy_sources = set(ALIASES)
    accuracy_sources.update(
        s for i, s in enumerate(data_sources) if isinstance(infos.get("acc", [None] * len(uids))[i], Real)
    )
    records = {}
    for i, (source, uid, index) in enumerate(zip(data_sources, uids, sample_indices, strict=True)):
        key = (str(source), str(uid))
        metadata = prompt_metadata[key]
        row = {k: v[i] for k, v in infos.items() if not k.startswith("__")}
        main = "acc" if source in accuracy_sources else "reward"
        score = row.get(main)
        status = "ok" if _finite(score) else ("score_nonfinite" if isinstance(score, Real) else "score_missing")
        row.update(
            data_source=source,
            prompt_id=metadata["prompt_id"],
            metric_fields=metadata.get("metric_fields", []),
            sample_index=int(index),
            uid=f"{uid}_{index}",
            input=inputs[i],
            output=outputs[i],
            gts=ground_truths[i],
            score=scores[i],
            response_length=response_lengths[i],
            num_turns=turns[i],
            status=status,
        )
        if (key, int(index)) in records:
            raise ValueError("Duplicate final evaluation session")
        records[(key, int(index))] = row
    for key, count in expected_counts.items():
        source, uid = key
        metadata = prompt_metadata[key]
        for index in range(count):
            records.setdefault(
                (key, index),
                dict(
                    data_source=source,
                    prompt_id=metadata["prompt_id"],
                    metric_fields=metadata.get("metric_fields", []),
                    sample_index=index,
                    uid=f"{uid}_{index}",
                    input="",
                    output=None,
                    prompt=metadata.get("prompt"),
                    gts=metadata.get("ground_truth"),
                    score=None,
                    reward=None,
                    acc=None,
                    pred=None,
                    response_length=None,
                    num_turns=None,
                    status="generation_missing",
                ),
            )
    return sorted(records.values(), key=lambda r: (r["data_source"], r["prompt_id"], r["sample_index"]))
