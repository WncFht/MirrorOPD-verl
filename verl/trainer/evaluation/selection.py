# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Outcome-independent question identities, stratified selection and source splits."""

import re
import unicodedata
from collections import defaultdict

from .artifacts import identity


def canonical_text(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def fingerprints(text):
    normalized = canonical_text(text)
    # Numeric variants remain in one family. This is a syntactic family split,
    # not a claim that semantic paraphrases have all been removed.
    family = re.sub(r"\d+(?:\.\d+)?", "#", normalized)
    return identity(normalized), identity(family)


def stratified_sample(rows, count, seed, *, stratum="stratum", namespace="selection"):
    rows = list(rows)
    if count > len(rows) or count < 0 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Selection requires unique identities and sufficient population")
    if not count:
        return []
    groups = defaultdict(list)
    for row in rows:
        groups[row[stratum]].append(row)
    quotas = {key: count * len(values) // len(rows) for key, values in groups.items()}
    remaining = count - sum(quotas.values())
    # Largest remainder preserves population proportions; hash breaks exact ties.
    ranked = sorted(groups, key=lambda key: (-(count * len(groups[key]) % len(rows)), identity([seed, namespace, key])))
    for key in ranked[:remaining]:
        quotas[key] += 1
    selected = [
        row
        for key, values in groups.items()
        for row in sorted(values, key=lambda r: identity([seed, namespace, r["id"]]))[: quotas[key]]
    ]
    return sorted(selected, key=lambda r: identity([seed, namespace, "order", r["id"]]))


def source_split(rows, protected, *, train_count=6400, heldout_count=128, seed=42):
    protected_families = {r["family"] for r in protected}
    groups = {}
    removed = 0
    for row in rows:
        if row["family"] in protected_families:
            removed += 1
            continue
        old = groups.get(row["family"])
        if old is None or row["id"] < old["id"]:
            groups[row["family"]] = row
    unique = list(groups.values())
    heldout = stratified_sample(unique, heldout_count, seed, namespace="source-heldout")
    heldout_families = {r["family"] for r in heldout}
    train = stratified_sample(
        [r for r in unique if r["family"] not in heldout_families], train_count, seed, namespace="training"
    )
    return (
        train,
        heldout,
        {
            "eligible_families": len(unique),
            "protected_removed": removed,
            "training": len(train),
            "heldout": len(heldout),
        },
    )
