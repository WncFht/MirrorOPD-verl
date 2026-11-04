# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate frozen E39/E41/E43 identities and derive figure labels from evidence.

This reader serves three historical runs. The training config is authoritative;
independent teacher metrics retain their own receipt, protocol, seed and samples.
No trainer, GPU, W&B API or weekly-report input is needed.
"""

import gzip
import hashlib
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from examples.mirroropd.research.core.artifacts import fingerprint
from examples.mirroropd.research.core.paths import REPO

DEFAULT_MANIFEST = (
    REPO / "examples/mirroropd/docs/experiments/teacher-transfer/data/qwen-training-figures/manifest.json"
)
COUNTS = {"math500": 500, "amc23": 40, "aime24": 30, "aime25": 30, "aime26": 30}


def checked_json(path, expected_sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"Frozen figure input hash mismatch: {path}")
    return json.loads(raw)


def config_protocol(config):
    """Extract actual model, loss, training and decoding settings without defaults."""
    actor = config["actor_rollout_ref"]["actor"]
    rollout = config["actor_rollout_ref"]["rollout"]
    loss = config["distillation"]["distillation_loss"]
    mirror = loss["mirror"]
    if (loss["loss_mode"], mirror["objective"], mirror["estimator"], mirror["kappa"], mirror["eta"]) != (
        "mirror",
        "mdpo",
        "full_vocab",
        0,
        1,
    ):
        raise ValueError("These historical figures require Full-RKL (kappa=0, eta=1)")
    val = rollout["val_kwargs"]
    thinking = config["data"]["apply_chat_template_kwargs"]["enable_thinking"]
    return {
        "student": Path(config["actor_rollout_ref"]["model"]["path"]).name,
        "teacher": Path(config["distillation"]["teacher_models"]["teacher_model"]["model_path"]).name,
        "enable_thinking": thinking,
        "training": {
            "objective": "Full-RKL",
            "reuse": actor["ppo_epochs"],
            "batch": config["data"]["train_batch_size"],
            "learning_rate": actor["optim"]["lr"],
            "samples": rollout["n"],
            "temperature": rollout["temperature"],
            "top_p": rollout["top_p"],
            "response_cap": rollout["response_length"],
            "seed": rollout["seed"],
            "termination_token_ids": loss["full_vocab"]["termination_token_ids"],
            "task_rewards_in_loss": loss["use_task_rewards"],
        },
        "evaluation": {
            "metric": "mean_correctness",
            "samples": val["n"],
            "temperature": val["temperature"],
            "top_p": val["top_p"],
            "top_k": val["top_k"],
            "response_cap": val["max_tokens"],
            "enable_thinking": thinking,
            "seed": rollout["seed"],
            "initial_validation": config["trainer"]["val_before_train"],
            "question_sets": [Path(p).parent.name for p in config["data"]["val_files"]],
        },
    }


def teacher_reference(evidence, directory, student):
    """Reject changed/incomplete teacher evidence and incompatible decoding settings."""
    loaded = {key: checked_json(directory / item["path"], item["sha256"]) for key, item in evidence.items()}
    metrics, complete, protocol = (loaded[k] for k in ["metrics", "complete", "protocol"])
    inputs = complete["inputs"]
    if complete["fingerprint"] != fingerprint(inputs) or inputs["protocol"] != fingerprint(protocol):
        raise ValueError("Teacher completion receipt does not match its protocol")
    if complete["metrics_sha256"] != evidence["metrics"]["sha256"]:
        raise ValueError("Teacher completion receipt does not match its metrics")
    accuracy = protocol["accuracy"]
    profile = inputs["profile"]
    decoding = {
        "metric": "mean_correctness",
        "samples": profile["samples"],
        "temperature": inputs["temperature"],
        "top_p": inputs["top_p"],
        "response_cap": profile["response_cap"],
        "seed": profile["seed"],
        "enable_thinking": protocol["chat_template_kwargs"]["enable_thinking"],
        "stop_ids": accuracy["stop_ids"],
    }
    for key in ["samples", "temperature", "top_p", "response_cap", "seed"]:
        if decoding[key] != accuracy[key]:
            raise ValueError(f"Teacher receipt/protocol mismatch: {key}")
    if protocol["models"]["Q"] != student["teacher"]:
        raise ValueError("Teacher model does not match the training pair")
    for key in ["metric", "temperature", "top_p", "response_cap", "enable_thinking"]:
        if decoding[key] != student["evaluation"][key]:
            raise ValueError(f"Teacher/student evaluation protocol mismatch: {key}")
    n = decoding["samples"]
    if (
        complete["samples"] != n
        or inputs["smoke"]
        or complete["questions"] != sum(COUNTS.values())
        or complete["records"] != sum(COUNTS.values()) * n
    ):
        raise ValueError("Teacher full-set completion count mismatch")
    rows = []
    for bench, count in COUNTS.items():
        prefix = f"eval/{bench}/"
        expected = {
            "prompt_count": count,
            "response_count": count * n,
            "observed_response_count": count * n,
            "failed_count": 0,
            "nonfinite_score_count": 0,
            "materialized_response_count": count * n,
            "samples_per_prompt_min": n,
            "samples_per_prompt_max": n,
        }
        if any(metrics[prefix + key] != value for key, value in expected.items()):
            raise ValueError(f"Incomplete teacher benchmark: {bench}")
        score = metrics[prefix + "score/mean"]
        if score != metrics[prefix + "acc/mean"] or not 0 <= score <= 1:
            raise ValueError(f"Teacher mean correctness mismatch: {bench}")
        rows.append({"task": bench, "accuracy": score, "samples": n, "question_count": count})
    return {"rows": rows, "evaluation": decoding, "model": protocol["models"]["Q"], "evidence": evidence}


def load_inputs(tag, config_path, manifest_path=DEFAULT_MANIFEST):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != 1:
        raise ValueError("Unsupported Qwen figure input schema")
    entry = manifest["runs"][tag]
    config = checked_json(config_path, entry["config"]["sha256"])
    protocol = config_protocol(config)
    key = entry["teacher_full_set"]
    teacher = teacher_reference(manifest["teacher_evidence"][key], manifest_path.parent, protocol) if key else None
    return {
        "experiment": tag,
        **entry,
        **protocol,
        "teacher_reference": teacher,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "reader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def check_source(path, entry):
    raw = Path(path).read_bytes()
    if entry.get("hash_scope") == "uncompressed_bytes":
        raw = gzip.decompress(raw)
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"Frozen plot input hash mismatch: {path}")


def check_plot_inputs(inputs, directory):
    """Bind the plotting tables/file logger to frozen evidence, never rewrite them."""
    for name, entry in inputs["plot_inputs"].items():
        check_source(directory / name, entry)


def check_decoded_inputs(inputs, directory):
    # gzip headers contain decode timestamps; bind decompressed telemetry bytes.
    for name, entry in inputs["decoded_inputs"].items():
        check_source(directory / name, entry)


def percent_label(value):
    """Round displayed accuracy percentages half up; retain raw values in inputs."""
    return str((Decimal(str(value)) * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def metric_label(protocol):
    if protocol["metric"] != "mean_correctness":
        raise ValueError("Accuracy figures require mean correctness, not pass@k")
    return f"avg@{protocol['samples']}"


def setting_label(inputs):
    train = inputs["training"]
    thinking = "ON" if inputs["enable_thinking"] else "OFF"
    learning_rate = f"{train['learning_rate']:g}".replace("e-0", "e-")
    return (
        f"{inputs['experiment']}: Student {inputs['student']}  ←  Teacher {inputs['teacher']}\n"
        f"Thinking {thinking} (enable_thinking={inputs['enable_thinking']}); {train['objective']}, "
        f"reuse{train['reuse']}, batch{train['batch']}, LR {learning_rate}"
    )


def decoding_label(protocol):
    thinking = "ON" if protocol["enable_thinking"] else "OFF"
    return (
        f"thinking {thinking}; T={protocol['temperature']:g}, top-p={protocol['top_p']:g}; "
        f"response cap={protocol['response_cap']:,} tokens."
    )
