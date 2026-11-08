# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit and reaggregate existing S0 full-vocabulary parent/descendant forwards.

No model inference or training. KL directions use the same frozen S0 prefixes;
an available fkl_shared is a full reverse direction only after support checks.
Question quantiles summarize question means, never individual token positions.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from examples.mirroropd.assets.paths import available_model, model_entry
from examples.mirroropd.research.core.paths import REPO
from examples.mirroropd.research.studies.fixed_transfer.figure_data import DOMAINS, GROUPS, HELD, registry

BASE = REPO / "examples/mirroropd/docs/experiments/teacher-transfer"
CATALOG = BASE / "transfer-mechanism/data/overview-20261029"
OUT = BASE / "fixed-state-results/figures/rl-parent-kl-20261108"
SELECTION = {
    "r1": ("deepscaler", "deepcoder", "justrl", "nemotron", "brorl", "minimath"),
    "q3-1.7": ("grpo-kl", "e3", "sft-gsm8k"),
    "llama3-3b-off": ("gt", "coreward-i", "arithmetic-full"),
}
ALIASES = {
    "r1": "r1",
    "q3-1.7": "Qwen3-1.7B",
    "llama3-3b-off": "Llama-3.2-3B-Instruct",
    "deepscaler": "deepscaler",
    "deepcoder": "deepcoder",
    "justrl": "justrl",
    "nemotron": "nemotron",
    "brorl": "brorl",
    "minimath": "minimath",
    "grpo-kl": "Qwen3-1.7B-GRPO-KL-math-reasoning",
    "e3": "e3-1.7B",
    "sft-gsm8k": "Qwen3-1.7B-Math",
    "gt": "GT-Llama-3.2-3B-Instruct-MATH",
    "coreward-i": "Co-rewarding-I-Llama-3.2-3B-Instruct-MATH",
    "arithmetic-full": "Llama-3.2-3B-Instruct-arithmetic-full",
}
METRICS = {"pre_to_post": "rkl", "post_to_pre": "fkl_shared", "js": "js", "tv": "tv"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def support(key):
    alias = ALIASES[key]
    entry = model_entry(alias)
    root = available_model(alias)
    config, tokenizer = root / "config.json", root / "tokenizer.json"
    vocab_size = json.loads(config.read_text())["vocab_size"]
    tok = json.loads(tokenizer.read_text())
    # Compare token IDs/content, not tokenizer file hashes (templates can differ).
    vocabulary = dict(tok["model"]["vocab"])
    for added in tok["added_tokens"]:
        content, index = added["content"], added["id"]
        if content in vocabulary and vocabulary[content] != index:
            raise ValueError(f"Conflicting token ID: {key}/{content}")
        vocabulary[content] = index
    fingerprint = hashlib.sha256(json.dumps(vocabulary, sort_keys=True).encode()).hexdigest()
    return {
        "repo": entry["repo"],
        "revision": entry["revision"],
        "head_vocab_size": vocab_size,
        "token_ids_sha256": fingerprint,
        "config": str(config),
        "config_sha256": digest(config),
        "tokenizer": str(tokenizer),
        "tokenizer_sha256": digest(tokenizer),
    }


def summarize(rows, student, teacher, panel, band):
    stats = [row["stats"][band] for row in rows if row["stats"][band]["tokens"] > 0]
    if not stats:
        return None
    tokens = sum(row["tokens"] for row in stats)
    record = dict(
        student=student,
        teacher=teacher,
        stage="SFT control"
        if teacher in {"minimath", "sft-gsm8k", "arithmetic-full"}
        else "RL cumulative"
        if teacher == "brorl"
        else "RL",
        panel=panel,
        band=band,
        questions=len(rows),
        nonempty_questions=len(stats),
        tokens=tokens,
        horizon=1024,
        coverage=tokens / (len(rows) * 1024) if band == "all" else None,
    )
    for name, key in METRICS.items():
        present = [key in row for row in stats]
        if any(present) != all(present):
            raise ValueError(f"Partially missing metric: {student}/{teacher}/{key}")
        record[name] = math.fsum(row[key] for row in stats) / tokens if all(present) else None
    question_means = [row["rkl"] / row["tokens"] for row in stats]
    record.update(
        pre_to_post_question_mean=float(np.mean(question_means)),
        pre_to_post_q10=float(np.quantile(question_means, 0.1)),
        pre_to_post_q50=float(np.quantile(question_means, 0.5)),
        pre_to_post_q90=float(np.quantile(question_means, 0.9)),
        fixed_h_pre_to_post=math.fsum(row["rkl"] for row in stats) / (len(rows) * 1024) if band == "all" else None,
    )
    if band == "all":
        assert math.isclose(record["fixed_h_pre_to_post"], record["pre_to_post"] * record["coverage"], rel_tol=1e-12)
    return record


def main():
    specs = {(r["student"], r["teacher"]): r for r in registry(CATALOG)}
    assets = {key: support(key) for key in ALIASES}
    sources, output, questions = [], [], []
    identities = {}
    with (BASE / "figures/fixed-state-clean-20261107/endpoints.csv").open() as stream:
        endpoints = {
            (r["student"], r["teacher"]): float(r["fixed_h_initial"])
            for r in csv.DictReader(stream)
            if r["panel"] == "All held-out" and r["band"] == "all"
        }
    for student, teachers in SELECTION.items():
        for teacher in teachers:
            spec = specs[student, teacher]
            path = Path(spec["initial_source"])
            rows = json.loads(path.read_text())
            if len({r["id"] for r in rows}) != len(rows) or any(r["step"] != 0 for r in rows):
                raise ValueError(f"Invalid S0 snapshot: {path}")
            if not set(HELD) <= {r["group"] for r in rows}:
                raise ValueError(f"Incomplete held-out snapshot: {path}")
            for r in rows:
                a = r["stats"]["all"]
                assert 0 < a["tokens"] == r["length"] <= 1024
                for metric in METRICS.values():
                    if metric in a:
                        assert math.isfinite(a[metric]) and a[metric] >= -1e-6
                assert math.isclose(
                    sum(r["stats"][b]["rkl"] for b in ["early", "middle", "late"]), a["rkl"], abs_tol=1e-5
                )
                assert sum(r["stats"][b]["tokens"] for b in ["early", "middle", "late"]) == a["tokens"]
                if "extra_teacher_mass" in a:
                    assert a["extra_teacher_mass"] == 0
            for k in ["head_vocab_size", "token_ids_sha256"]:
                if assets[student][k] != assets[teacher][k]:
                    raise ValueError(f"Unequal native support: {student}/{teacher}/{k}")
            held = [r for r in rows if r["group"] in HELD]
            identity = sorted((r["id"], r["group"], r["source"], r["length"]) for r in held)
            if student in identities and identity != identities[student]:
                raise ValueError(f"Held-out state metadata changed: {student}")
            identities[student] = identity
            panels = {"All held-out": HELD, **DOMAINS, "Training scored": GROUPS}
            for panel, groups in panels.items():
                chosen = [r for r in rows if r["group"] in groups]
                for band in ["all", "early", "middle", "late"]:
                    result = summarize(chosen, student, teacher, panel, band)
                    if result:
                        output.append(result)
                        if panel == "All held-out" and band == "all":
                            assert len(chosen) == 128
                            assert math.isclose(
                                result["fixed_h_pre_to_post"], endpoints[student, teacher], abs_tol=1e-12
                            )
            for r in held:
                a = r["stats"]["all"]
                questions.append(
                    dict(
                        student=student,
                        teacher=teacher,
                        id=r["id"],
                        group=r["group"],
                        tokens=a["tokens"],
                        pre_to_post_sum=a["rkl"],
                        post_to_pre_sum=a.get("fkl_shared"),
                        pre_to_post=a["rkl"] / a["tokens"],
                    )
                )
            sources.append(
                dict(
                    student=student,
                    teacher=teacher,
                    path=str(path),
                    sha256=digest(path),
                    registry_run_id=spec["run_id"],
                )
            )
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "parent-kl.csv", output)
    write_csv(OUT / "questions.csv", questions)
    mimo_source = BASE / "data/mimo-stage-20261104-final/E72/fixed-bank-rkl.csv"
    frozen_manifest = BASE / "data/mimo-position-kl-20261106/manifest.json"
    assert (
        digest(mimo_source)
        == json.loads(frozen_manifest.read_text())["sources"][mimo_source.relative_to(REPO).as_posix()]
    )
    with mimo_source.open() as stream:
        mimo = [
            dict(r, direction="KL(MiMo-Base || MiMo-RL-Zero)", aggregation="mean of question token means")
            for r in csv.DictReader(stream)
            if r["optimizer_updates"] == "0"
        ]
    assert len(mimo) == 10
    write_csv(OUT / "mimo-parent-kl.csv", mimo)
    provenance = dict(
        method="reaggregate original S0 full-vocabulary forwards; no new inference",
        units="nats per observed token; TV is dimensionless",
        state_distribution="frozen initial student prefixes; identical within each student",
        post_to_pre=(
            "fkl_shared only for verified identical native vocabulary; same initial prefixes, not post-RL rollouts"
        ),
        question_quantiles="quantiles of per-question token mean, not per-token KL quantiles",
        horizon=1024,
        heldout_groups=HELD,
        assets=assets,
        sources=sources,
        mimo_source=dict(path=str(mimo_source), sha256=digest(mimo_source)),
        registry_sources=[
            dict(path=str(CATALOG / n), sha256=digest(CATALOG / n)) for n in ["risks.csv", "appendix-risks.csv"]
        ],
        script_sha256=digest(Path(__file__)),
        scoring_code_sha256=digest(REPO / "examples/mirroropd/research/core/metrics.py"),
    )
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for r in output:
        if r["panel"] == "All held-out" and r["band"] == "all":
            print(json.dumps(r))
    print(
        f"Validated {len(sources)} pair snapshots, {len(questions)} held-out question records, "
        f"{len(output)} panel/band summaries and {len(mimo)} MiMo bank summaries"
    )


if __name__ == "__main__":
    main()
