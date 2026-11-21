# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read measured artifacts and resolved settings without recomputing state geometry."""

from pathlib import Path

import numpy as np
import yaml

from examples.mirroropd.research.core.io import read

from .artifacts import cached, digest
from .overview import teacher_overview

GROUP_LABELS = {
    "initial": "Initial 回答",
    "outer01_10": "训练阶段 01–10",
    "outer11_20": "训练阶段 11–20",
    "outer21_30": "训练阶段 21–30",
    "outer31_40": "训练阶段 31–40",
    "teacher": "Teacher 回答",
}
COLORS = ["#85898F", "#4F8FA5", "#EE995B", "#C95B5B", "#8A74B5", "#3A3F45"]


def model_name(path):
    directory = Path(path)
    return directory.parent.name.replace("--", "/") if len(directory.name) >= 32 else directory.name.replace("--", "/")


def risk_rows(plan):
    """Load all recorded outer evaluations, retaining bank/domain and source receipts."""
    result, sources = [], []
    for pair, spec in plan["pairs"].items():
        manifests = sorted((Path(spec["evaluation"]) / "events").glob("outer-*/manifest.json"))
        events = [read(path)["metadata"] for path in manifests]
        sources.extend(dict(path=str(path), sha256=digest(path)) for path in manifests)
        if not events:
            events = [event["metadata"] for event in spec["risks"]]
        for event in events:
            for bank in ("public_initial", "public_teacher"):
                for domain in ("math", "code", "if", "all"):
                    prefix = f"eval/rkl/{bank}/{domain}/"
                    metrics = event["metrics"]
                    if prefix + "question_mean" not in metrics:
                        continue
                    result.append(
                        dict(
                            pair=pair,
                            outer=event["outer"],
                            steps=event["optimizer_steps"],
                            bank=bank,
                            domain=domain,
                            rkl=metrics[prefix + "question_mean"],
                            reduction_percent=metrics[prefix + "reduction_percent"],
                            rows=metrics[prefix + "rows"],
                            tokens=metrics[prefix + "tokens"],
                        )
                    )
    return result, sources


def pair_payload(plan_path, pair, spec, groups):
    root = plan_path.parent
    config_path = Path(spec["config"])
    if digest(config_path) != spec["config_sha256"]:
        raise ValueError(f"Resolved training configuration changed: {config_path}")
    config = yaml.safe_load(config_path.read_text())
    actor, data = config["actor_rollout_ref"]["actor"], config["data"]
    optim = actor["optim"]
    result_path = root / "analysis" / pair / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Complete the requested state analysis first: {result_path}")
    result = read(result_path)
    points = result["map"]["points"]
    texts = {}
    for point in points:
        meta = point["metadata"]
        entry = texts.setdefault(point["trajectory"], dict(prompt=meta.pop("prompt_text", ""), contexts={}))
        meta.pop("prompt_text", None)
        entry["contexts"][str(point["order"])] = meta.pop("context", "")
    stats = {}
    for group in groups:
        selected = [p for p in points if p["group"] == group]
        trajectories = {p["trajectory"]: p for p in selected}
        stats[group] = dict(
            trajectories=len(trajectories),
            states=len(selected),
            metric_states=sum(p["metadata"]["action"] > 0 for p in selected),
            mean_tokens=sum(p["metadata"]["response_tokens"] for p in trajectories.values()) / len(trajectories),
            test_trajectories=sum(p["metadata"]["split"] in ("test", "query") for p in trajectories.values()),
        )
    signatures, source_files = [], {}
    for task in spec["tasks"]:
        path = root / "signatures" / pair / f"{task['id']}.npz"
        receipt = read(path.with_suffix(".json"))
        if not cached(path, receipt["key"]):
            raise FileNotFoundError(path)
        signatures.append(receipt)
        source_files[task["source"]] = task["source_sha256"]
    signature_hashes = [r["sha256"] for r in signatures]
    if result["metrics"]["provenance"]["signature_sha256"] != signature_hashes:
        raise ValueError(f"Analysis and signature sources differ: {pair}")
    with np.load(root / "analysis" / pair / "geometry.npz", allow_pickle=False) as geometry:
        reference = geometry["reference_indices"]
        reference_trajectories = {points[i]["trajectory"] for i in reference}
    before, after = config["trainer"]["total_training_steps"], actor["ppo_epochs"]
    settings = dict(
        student=model_name(config["actor_rollout_ref"]["model"]["path"]),
        teacher=model_name(spec["teacher"]["path"]),
        optimizer=optim["optimizer"],
        lr=optim["lr"],
        weight_decay=optim["weight_decay"],
        betas=optim["betas"],
        clip_grad=optim["clip_grad"],
        schedule=optim["lr_scheduler_type"],
        outer_steps=before,
        reuse=after,
        updates=before * after,
        batch=data["train_batch_size"],
        training_questions=before * data["train_batch_size"],
        seed=data["seed"],
        prompt_cap=data["max_prompt_length"],
        response_cap=data["max_response_length"],
        template=data["apply_chat_template_kwargs"],
        family=data["continuous_token"]["model_family"],
        temperature=config["actor_rollout_ref"]["rollout"]["temperature"],
        loss=config["distillation"]["distillation_loss"]["mirror"],
        rollout=config["rollout_policy"]["model"]["kind"],
        accuracy_measured=False,
    )
    elapsed = sum(r["elapsed_seconds"] for r in signatures)
    extraction = dict(
        tokens=sum(r["tokens"] for r in signatures),
        forward_seconds=elapsed,
        tokens_per_second=sum(r["tokens"] for r in signatures) / elapsed,
        peak_gib=max(r["peak_allocated_bytes"] for r in signatures) / 2**30,
        alignment_cosine=min(r["alignment_cosine"] for r in signatures),
        settings=signatures[0]["settings"],
    )
    available_training = sum(
        read(Path(spec["evaluation"]) / "manifest.json")["files"][f"trajectories/outer-{o:03d}.parquet"]["rows"]
        for o in range(1, 41)
    )
    return dict(
        settings=settings,
        points=points,
        trajectories=texts,
        metrics=result["metrics"],
        overview=teacher_overview(points, groups, seed=read(plan_path)["seed"]),
        stats=stats,
        extraction=extraction,
        reference_trajectories=len(reference_trajectories),
        available_training=available_training,
        selected_training=sum(stats[g]["trajectories"] for g in groups[1:-1]),
        groups=[dict(id=g, label=GROUP_LABELS.get(g, g), color=c) for g, c in zip(groups, COLORS, strict=True)],
        provenance=dict(
            config=str(config_path),
            config_sha256=spec["config_sha256"],
            teacher=spec["teacher"],
            result_sha256=digest(result_path),
            source_files=source_files,
            analysis=result["metrics"]["provenance"],
        ),
    )


def payload(plan_path):
    plan_path = Path(plan_path)
    plan = read(plan_path)
    rows, events = risk_rows(plan)
    result = dict(
        schema="latent-state-report-v2",
        groups=plan["groups"],
        seed=plan["seed"],
        per_outer=plan["per_outer"],
        pairs={pair: pair_payload(plan_path, pair, spec, plan["groups"]) for pair, spec in plan["pairs"].items()},
        risks=rows,
        provenance=dict(plan_sha256=digest(plan_path), evaluation_events=events),
    )
    audit_path = plan_path.parent / "prompt-audit.json"
    if audit_path.exists():
        audit = read(audit_path)
        for pair, spec in plan["pairs"].items():
            config = yaml.safe_load(Path(spec["config"]).read_text())
            cfg = read(Path(spec["evaluation"]) / "manifest.json")["metadata"]["configuration"]
            math_path = next(d["path"] for d in cfg["datasets"] if d["name"] == "math")
            if audit[pair]["train_sha256"] != digest(config["data"]["train_files"][0]) or audit[pair][
                "math_sha256"
            ] != digest(math_path):
                raise ValueError(f"Prompt overlap audit no longer matches data: {pair}")
            result["pairs"][pair]["prompt_audit"] = audit[pair]
    return result
