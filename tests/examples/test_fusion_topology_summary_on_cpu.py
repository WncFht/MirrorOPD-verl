# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Topology evidence must match the declared physical allocation and mixture."""

import json

import pytest

from examples.mirroropd.benchmarks.fusion.summarize_topology import summarize


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def run(tmp_path):
    config = dict(
        tp=1,
        replicas=2,
        batch=2,
        cap=2,
        repeats=1,
        alpha=0.5,
        student="s",
        teacher="t",
        physical_gpus="4,5",
        fusion_space="probability",
    )
    write(tmp_path / "config.json", config)
    write(tmp_path / "prompts.json", [{"prompt_token_ids": [11]}, {"prompt_token_ids": [12]}])
    write(tmp_path / "source.json", {"git_head": "fixture"})
    batches = []
    for repeat in [-1, 0]:
        release = 10.0 * (repeat + 2)
        ended, collected = release + 1.0, release + 1.2
        write(tmp_path / f"go-{repeat}.json", {"released_monotonic": release})
        replicas = []
        for rank in [0, 1]:
            replicas.append(
                dict(
                    replica=rank,
                    started_monotonic=release + 0.1 * (rank + 1),
                    ended_monotonic=ended,
                    responses=[
                        dict(
                            global_index=rank,
                            seed=42 + rank,
                            prompt_ids=[11 + rank],
                            token_ids=[151643],
                            log_probs=[-0.2],
                            finish_reason="stop",
                        )
                    ],
                    backend_metrics_cumulative=[dict(name="vllm:num_preemptions", value=0)],
                )
            )
        batches.append(
            dict(
                repeat=repeat,
                seconds=collected - release,
                collected_monotonic=collected,
                engine_completion_seconds=ended - release,
                result_collection_seconds=collected - ended,
                launch_skew_seconds=replicas[1]["started_monotonic"] - replicas[0]["started_monotonic"],
                replicas=replicas,
            )
        )
    write(tmp_path / "results.json", {"batches": batches})
    for rank in [0, 1]:
        write(
            tmp_path / f"replica-{rank}-audit.json",
            [
                dict(
                    tp_rank=0,
                    tensor_parallel_size=1,
                    alpha=0.5,
                    fusion_space="probability",
                    student_path="s",
                    teacher_path="t",
                )
            ],
        )
    resources = [
        dict(kind="metadata", physical_gpus=[4, 5]),
        dict(
            kind="sample",
            gpus=[dict(index=4, memory_used=100), dict(index=5, memory_used=200)],
            process_tree_rss_sum=300,
        ),
        dict(kind="end", wall_seconds=10),
    ]
    (tmp_path / "resources.jsonl").write_text("\n".join(json.dumps(r) for r in resources))
    return tmp_path


def test_summary_uses_declared_physical_pair(run):
    result = summarize(run)
    assert result["nvml_peak_bytes_by_gpu"] == {"4": 100, "5": 200}
    assert result["formal_generation_complete"] is False


def test_summary_rejects_another_pairs_resource_trace(run):
    path = run / "resources.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["physical_gpus"] = [6, 7]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="resource interval"):
        summarize(run)


def test_summary_rejects_wrong_fusion_formula(run):
    path = run / "replica-1-audit.json"
    rows = json.loads(path.read_text())
    rows[0]["fusion_space"] = "logit"
    write(path, rows)
    with pytest.raises(ValueError, match="fusion space"):
        summarize(run)
