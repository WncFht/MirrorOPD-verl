# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Scientific budgets, frozen question ordering and non-interfering GPU admission."""

import json
from types import SimpleNamespace

import pytest
from hydra.core.override_parser.overrides_parser import OverridesParser

from examples.mirroropd.launchers import math_experiment as planner
from examples.mirroropd.launchers.qwen_prefix import SOURCE_PROTOCOL, evaluation
from examples.mirroropd.research.studies.prefix_fusion.native_source_queue import choose_devices, reserve_all
from verl.trainer.evaluation.config import EvaluationConfig
from verl.trainer.evaluation.plan import compile_plan


@pytest.mark.parametrize("number,kind", [(78, "current"), (79, "initial"), (80, "teacher")])
@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_new_presets_have_one_global_budget_and_independent_bench_n(number, kind, world, tmp_path):
    args, rest = planner.parser().parse_known_args([f"E{number}"])
    plan = planner.build_plan(
        args,
        rest,
        {
            "RUN_DIR": str(tmp_path / "run"),
            "STUDENT_GPUS": str(world),
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(world))),
        },
    )
    assert plan.environment["TRAIN_BATCH_SIZE"] == "128"
    assert plan.environment["PPO_EPOCHS"] == "16"
    assert "trainer.total_training_steps=100" in plan.arguments
    assert f"rollout_policy.model={{kind:{kind}}}" in plan.arguments
    assert "data.shuffle=False" in plan.arguments
    assert "data.filter_overlong_prompts=False" in plan.arguments
    assert "rollout_policy.execution.tensor_parallel_size=1" in plan.arguments
    encoded = next(
        a.removeprefix("++trainer.evaluation=") for a in plan.arguments if a.startswith("++trainer.evaluation=")
    )
    value = OverridesParser.create().parse_override("trainer.evaluation=" + encoded).value()
    value.pop("_target_")
    options = EvaluationConfig.parse(value)
    assert options.source_enabled and not options.history_enabled
    assert options.source_count == 128
    assert sum(len(d.panel_ids) for d in options.datasets) == 128
    assert {d.name: d.n for d in options.datasets} == {
        "MATH-500": 4,
        "AMC23": 16,
        "AIME24": 32,
        "AIME25": 32,
        "AIME26": 32,
    }
    assert [
        n for n in range(101) if "accuracy" in compile_plan(options, outer=n, total=100, optimizer_steps=n * 16).tasks
    ] == [0, 50, 100]
    assert all(
        "history" not in compile_plan(options, outer=n, total=100, optimizer_steps=n * 16).tasks for n in range(101)
    )
    assert [
        n
        for n in range(101)
        if "source_heldout" in compile_plan(options, outer=n, total=100, optimizer_steps=n * 16).tasks
    ] == list(range(0, 101, 10))


def test_legacy_qwen_protocol_keeps_its_original_budget(tmp_path):
    old = evaluation(tmp_path, tmp_path / "out", 8)
    assert old["source_enabled"] and old["history_enabled"]
    assert old["source_count"] == 64
    assert {d["n"] for d in old["datasets"]} == {4}
    assert old["schedules"]["bench"]["at"] == [20, 40]
    new = evaluation(tmp_path, tmp_path / "out", 2, protocol=SOURCE_PROTOCOL)
    assert new["datasets"] == evaluation(tmp_path, tmp_path / "other", 1, protocol=SOURCE_PROTOCOL)["datasets"]
    assert all(len(set(d["panel_ids"])) == len(d["panel_ids"]) for d in new["datasets"])


class Broker:
    def __init__(self, available=()):
        self.data = {
            "gpus": {
                str(g): dict(
                    owner=None if g in available else "another-job",
                    desired="released" if g in available else "reserved",
                    active_run=f"occupy-{g}" if g in available else None,
                )
                for g in range(8)
            }
        }
        self.calls = []

    def registry(self):
        return self.data

    def owned(self, run, gpu):
        self.calls.append(("verify", gpu))

    def gpu_lock_free(self, gpu):
        return True

    def gpu_busy(self, gpu):
        return False

    def reserve(self, gpu, owner):
        entry = self.registry()["gpus"][str(gpu)]
        entry.update(owner=owner, desired="reserved", active_run=None)
        self.calls.append(("reserve", gpu))

    def release(self, gpu, owner, reason):
        assert self.data["gpus"][str(gpu)]["owner"] == owner
        self.data["gpus"][str(gpu)].update(owner=None, desired="released")
        self.calls.append(("release", gpu))


def test_partial_resources_start_without_stealing_or_waiting_for_eight():
    broker = Broker([0, 5, 6])
    assert choose_devices(broker) == (5, 6)
    assert reserve_all(broker, (5, 6))[0]
    assert broker.data["gpus"]["1"]["owner"] == "another-job"
    assert not any(action == "reserve" and gpu not in (5, 6) for action, gpu in broker.calls)
    assert choose_devices(Broker([0])) == (0,)
    assert choose_devices(Broker()) == ()


def test_unknown_compute_context_is_never_treated_as_idle():
    broker = Broker([5])
    broker.data["gpus"]["5"]["active_run"] = None
    broker.gpu_busy = lambda gpu: True
    assert choose_devices(broker) == ()
    assert not reserve_all(broker, (5,))[0]
    assert not any(action == "reserve" for action, _ in broker.calls)


def test_question_preparation_freezes_12800_distinct_prompts(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from examples.mirroropd.research.core.artifacts import file_hash
    from examples.mirroropd.research.studies.prefix_fusion.prepare_native import prepare

    def tokens(messages, **kwargs):
        assert kwargs["enable_thinking"] is False and kwargs["add_generation_prompt"]
        assert kwargs["return_dict"] is False
        return [1, 2, 3]

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: SimpleNamespace(apply_chat_template=tokens))
    source = tmp_path / "source"
    source.mkdir()
    pq.write_table(
        pa.Table.from_pylist([dict(prompt=[dict(role="user", content=f"train-{i}")], index=i) for i in range(12810)]),
        source / "train.parquet",
    )
    held = [
        dict(id=i, domain="math", primary=i < 64, messages=[dict(role="user", content=f"held-{i}")], ground_truth="0")
        for i in range(192)
    ]
    tasks = []
    for name, count in [("MATH-500", 500), ("AMC23", 40), ("AIME24", 30), ("AIME25", 30), ("AIME26", 30)]:
        tasks.extend(
            dict(data_source=name, messages=[dict(role="user", content=f"{name}-{i}")], ground_truth="0")
            for i in range(count)
        )
    (source / "heldout-questions.json").write_text(json.dumps(held))
    (source / "task-prompts.json").write_text(json.dumps(tasks))
    files = {p.name: file_hash(p) for p in source.iterdir()}
    (source / "complete.json").write_text(json.dumps(dict(artifacts=files)))
    for name in ["a", "b"]:
        prepare(source, tmp_path / name, protocol=SOURCE_PROTOCOL, tokenizer_path=tmp_path / "model")
    a = pq.read_table(tmp_path / "a/train.parquet")
    b = pq.read_table(tmp_path / "b/train.parquet")
    assert len(a) == 12800 and a.equals(b)
    assert len(set(a["index"].to_pylist())) == 12800
    assert pq.ParquetFile(tmp_path / "a/source.parquet").metadata.num_rows == 128
    assert json.loads((tmp_path / "a/complete.json").read_text())["max_prompt_tokens"] == 3
    with pytest.raises(FileExistsError):
        prepare(source, tmp_path / "a", protocol=SOURCE_PROTOCOL, tokenizer_path=tmp_path / "model")


def test_queue_keeps_the_virtualenv_interpreter_path(tmp_path, monkeypatch):
    import sys

    from examples.mirroropd.research.studies.prefix_fusion import native_source_queue as queue

    interpreter = tmp_path / "venv-python"
    interpreter.symlink_to(sys.executable)
    calls = []
    monkeypatch.setattr(queue, "prepare", lambda *values, **kwargs: calls.append(values))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "queue",
            "prepare",
            "--repo",
            str(tmp_path),
            "--source",
            str(tmp_path),
            "--study",
            str(tmp_path / "study"),
            "--python",
            str(interpreter),
        ],
    )
    queue.main()
    assert calls[0][-1] == interpreter


def test_reservation_race_rolls_back_only_our_cards():
    broker = Broker([5, 6])
    original = broker.reserve

    def reserve(gpu, owner):
        if gpu == 6:
            broker.data["gpus"]["6"].update(owner="new-other-job", desired="reserved")
        original(gpu, owner)

    broker.reserve = reserve
    ok, reason = reserve_all(broker, (5, 6))
    assert not ok and "contention" in reason
    assert broker.data["gpus"]["5"]["owner"] is None
    assert broker.data["gpus"]["6"]["owner"] == "new-other-job"
    assert ("release", 5) in broker.calls and ("release", 6) not in broker.calls


@pytest.mark.parametrize("failure", [None, "skipped", "missing", "corrupted"])
def test_training_audit_uses_verified_optimizer_artifacts_without_file_logger(tmp_path, failure):
    import pyarrow as pa

    from examples.mirroropd.research.studies.prefix_fusion.native_source_queue import audit_training
    from verl.trainer.evaluation.artifacts import ArtifactStore

    root = tmp_path / "gate/training/E78/evaluation"
    store = ArtifactStore.create(root, kind="unified_evaluation_run", metadata={})
    schema = pa.schema([("step", pa.int64()), ("metrics", pa.map_(pa.string(), pa.float64()))])
    for n in range(1, 3):
        if failure == "missing" and n == 2:
            continue
        metrics = {
            "step/optimizer": float(n),
            "optim/actor/update_applied": 1.0,
            "optim/actor/update_skipped": float(failure == "skipped"),
            "optim/actor/loss_nonfinite": 0.0,
            "optim/actor/loss": 0.3,
            "optim/actor/grad_norm_preclip": 1.2,
        }
        store.write_rows(
            f"updates/actor-outer-{n:03d}.parquet",
            [{"step": n, "metrics": list(metrics.items())}],
            schema=schema,
        )
    store.finish(metadata={"optimizer_steps": 2, "completed_outer": 2})
    export = root / "final-model/export-audit"
    export.mkdir(parents=True)
    (export / "manifest.json").write_text(json.dumps({"status": "complete"}))
    if failure == "corrupted":
        with (root / "updates/actor-outer-001.parquet").open("ab") as stream:
            stream.write(b"corrupted")
    if failure:
        with pytest.raises((ValueError, AssertionError)):
            audit_training(tmp_path, "E78", gate=True)
        assert not (tmp_path / "gate/E78-audit.json").exists()
    else:
        audit_training(tmp_path, "E78", gate=True)
        assert json.loads((tmp_path / "gate/E78-audit.json").read_text())["optimizer_updates"] == 2


def test_config_only_enables_local_metrics_and_creates_its_parent(tmp_path, monkeypatch):
    import yaml

    from examples.mirroropd.research.studies.prefix_fusion import native_source_queue as queue

    config = {"trainer": {"logger": ["console", "wandb"]}, "ray_kwargs": {"ray_init": {}}}
    monkeypatch.setattr(
        queue.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=yaml.safe_dump(config))
    )
    destination = tmp_path / "training/E78"
    resolved, _ = queue.config_only(tmp_path, tmp_path, tmp_path / "python", {}, "E78", destination)
    assert resolved["trainer"]["logger"] == ["console", "wandb", "file"]
    assert destination.is_dir()
    assert resolved["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]["VERL_FILE_LOGGER_PATH"] == str(
        destination / "metrics.jsonl"
    )
