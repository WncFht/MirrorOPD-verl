# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Disabled diagnostics do no model work and do not require a source bank."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.trainer.evaluation.artifacts import ArtifactStore
from verl.trainer.evaluation.config import EvaluationConfig
from verl.trainer.evaluation.coordinator import EvaluationCoordinator
from verl.trainer.evaluation.identity import active_bootstrap_contract, bootstrap_contract
from verl.trainer.evaluation.plan import compile_plan


def options(tmp_path, **changes):
    return EvaluationConfig.parse(
        {
            "output": str(tmp_path / "run"),
            "bootstrap": str(tmp_path / "bootstrap"),
            "datasets": [],
            "protocols": {"public": {}},
            **changes,
        }
    )


def test_default_off_including_initial_final_and_explicit_frequencies(tmp_path):
    cfg = options(tmp_path, schedules={"source": {"at": [0, 1, 50], "every": 1}, "history": {"every": 1}})
    for outer in range(51):
        tasks = compile_plan(cfg, outer=outer, total=50, optimizer_steps=outer * 32).tasks
        assert "source_heldout" not in tasks and "history" not in tasks
    assert not compile_plan(cfg, outer=3, total=50, optimizer_steps=96).tasks
    assert "source" not in cfg.bootstrap_children
    assert "source_sha256" not in bootstrap_contract(cfg, student="s", teacher="t")
    assert active_bootstrap_contract({"student": "s", "source_sha256": "old"}, source_enabled=False) == {"student": "s"}


def test_opt_in_restores_the_original_schedule(tmp_path):
    cfg = options(
        tmp_path,
        source_enabled=True,
        history_enabled=True,
        source_file="source",
        protocols={"source": {}, "public": {}},
    )
    assert "source" in cfg.bootstrap_children
    assert "source_heldout" in compile_plan(cfg, outer=0, total=50, optimizer_steps=0).tasks
    assert {"source_heldout", "history"} <= set(compile_plan(cfg, outer=3, total=50, optimizer_steps=96).tasks)
    with pytest.raises(ValueError, match="source_file"):
        options(tmp_path, source_enabled=True)
    with pytest.raises(ValueError, match="booleans"):
        options(tmp_path, history_enabled="false")


def test_off_skips_sampling_and_gpu_but_retains_training_tokens(tmp_path, monkeypatch):
    import verl.trainer.evaluation.coordinator as module

    coordinator = object.__new__(EvaluationCoordinator)
    coordinator.options = options(tmp_path)
    coordinator.config = OmegaConf.create({})
    coordinator.fixed = {"public_initial": [{"id": "fixed"}]}
    coordinator.batches = {}
    coordinator.history = []
    coordinator.completed = set()
    coordinator.hidden_size = 4
    coordinator.store = ArtifactStore.create(tmp_path / "run", kind="test", metadata={})
    monkeypatch.setattr(
        module,
        "capture_rows",
        lambda batch, outer: [
            dict(id=f"{outer}:q", source_id="q", outer=outer, prompt_ids=[1], response_ids=[2], response_mask=[True])
        ],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled history must not sample")

    monkeypatch.setattr(module, "balanced_history", forbidden)
    # No worker groups or rollout service: an empty evaluation boundary cannot touch them.
    trainer = SimpleNamespace(total_training_steps=50, actor_optimizer_step=96)
    assert coordinator.on_before_weight_sync(trainer, object(), 3) == {}
    assert 3 in coordinator.completed
    assert (tmp_path / "run/trajectories/outer-003.parquet").exists()
    assert not (tmp_path / "run/history").exists() and not (tmp_path / "run/events").exists()
    assert coordinator.checkpoint_references() == [{"id": "fixed"}]
    with pytest.raises(ValueError, match="repeated"):
        coordinator.on_before_weight_sync(trainer, object(), 4)


@pytest.mark.parametrize("enabled", [False, True])
def test_bootstrap_initial_only_generates_source_when_enabled(tmp_path, monkeypatch, enabled):
    import asyncio

    from verl.trainer.evaluation import bootstrap

    cfg = OmegaConf.create(
        {
            "trainer": {
                "n_gpus_per_node": 1,
                "evaluation": {
                    "_target_": "verl.trainer.evaluation.coordinator.EvaluationCoordinator",
                    "output": str(tmp_path / "run"),
                    "bootstrap": str(tmp_path / "bootstrap"),
                    "source_enabled": enabled,
                    "source_file": "unused-source",
                    "source_count": 1,
                    "datasets": [
                        {
                            "name": "math",
                            "domain": "math",
                            "path": "unused-public",
                            "verifier": "math",
                            "panel_ids": ["q"],
                        }
                    ],
                    "protocols": {"source": {"max_new_tokens": 8}, "public": {"max_new_tokens": 8}},
                },
            },
            "actor_rollout_ref": {"model": {"path": "student"}, "rollout": {}},
            "distillation": {"teacher_models": {"teacher_model": {"model_path": "teacher"}}},
        }
    )
    calls = []

    async def nothing(*args, **kwargs):
        return None

    class Replica:
        def __init__(self, *args, **kwargs):
            self.server_handle = SimpleNamespace(collective_rpc=SimpleNamespace(remote=nothing))
            self.servers = []
            self.workers = []
            self.resource_pool = None

        init_standalone = nothing

    class Generator:
        def __init__(self, *args, **kwargs):
            pass

        async def generate(self, destination, *args, **kwargs):
            calls.append(destination.name)
            return None

    class Collector:
        def __init__(self, *args, **kwargs):
            pass

        def follow_generation(self, *args):
            pass

        def submit(self, *args):
            pass

        def finish(self, *args):
            return {}

        def close(self):
            pass

    monkeypatch.setattr(bootstrap, "require_current_generation", lambda *a, **k: None)
    monkeypatch.setattr(
        bootstrap, "omega_conf_to_dataclass", lambda *a, **k: SimpleNamespace(path="student", tokenizer=None)
    )
    monkeypatch.setattr(bootstrap, "model_identity", lambda *a, **k: "m")
    monkeypatch.setattr(bootstrap, "prepared_generation_config", lambda *a: OmegaConf.create({}))
    monkeypatch.setattr(bootstrap, "get_rollout_replica_class", lambda name: Replica)
    monkeypatch.setattr(bootstrap, "NativeGenerator", Generator)
    monkeypatch.setattr(bootstrap, "AccuracyCollector", Collector)
    monkeypatch.setattr(
        bootstrap, "read_questions", lambda spec: [{"id": "q", "prompt": [{"role": "user", "content": "q"}]}]
    )
    asyncio.run(bootstrap.generate_bootstrap(cfg, "initial"))
    assert calls == (["initial", "source"] if enabled else ["initial"])


@pytest.mark.parametrize("enabled", [False, True])
def test_finalize_requires_only_enabled_bootstrap_children(tmp_path, monkeypatch, enabled):
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from verl.trainer.evaluation import bootstrap
    from verl.trainer.evaluation.artifacts import GENERATION, file_sha256, identity

    question = {"id": "q", "prompt": [{"role": "user", "content": "q"}]}
    path = tmp_path / "questions.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": "q", "prompt": json.dumps(question["prompt"])}]), path)
    values = dict(
        output=str(tmp_path / "out"),
        bootstrap=str(tmp_path / "boot"),
        source_enabled=enabled,
        source_file=str(path) if enabled else None,
        source_count=1,
        datasets=[dict(name="math", domain="math", path=str(path), verifier="math", panel_ids=["q"])],
        protocols={"public": {}, "source": {}},
    )
    opts = EvaluationConfig.parse(values)
    root = ArtifactStore.create(opts.bootstrap, kind="evaluation_bootstrap", metadata={})
    for child in ["initial", "teacher"] + (["source"] if enabled else []):
        name = "source_heldout" if child == "source" else "math"
        protocol = "source" if child == "source" else "public"
        store = ArtifactStore.create(
            root.root / child,
            kind="generation",
            metadata={
                "model_identity": "teacher" if child == "teacher" else "student",
                "engine_contract_id": identity({}),
                "protocols": {k: v.metadata() for k, v in opts.protocols.items()},
                "question_set_sha256": identity([(name, 1, protocol, [question])]),
            },
        )
        store.write_rows(
            "generations/part-00000.parquet",
            [
                dict(
                    id=child,
                    source_id="q",
                    outer=0,
                    prompt_ids=[1],
                    response_ids=[2],
                    response_mask=[True],
                    dataset=name,
                    domain="math",
                    sample_index=0,
                    valid_tokens=1,
                )
            ],
            schema=GENERATION,
        )
        store.finish()
    for child, producer in [("initial-accuracy", "initial"), ("teacher-accuracy", "teacher")]:
        store = ArtifactStore.create(
            root.root / child,
            kind="accuracy",
            metadata={"generation_sha256": file_sha256(root.root / producer / "manifest.json")},
        )
        store.finish()
    cfg = OmegaConf.create(
        {
            "trainer": {
                "evaluation": {"_target_": "verl.trainer.evaluation.coordinator.EvaluationCoordinator", **values}
            },
            "actor_rollout_ref": {"model": {"path": "student"}, "rollout": {}},
            "distillation": {"teacher_models": {"teacher_model": {"model_path": "teacher"}}},
        }
    )
    monkeypatch.setattr(bootstrap, "model_identity", lambda path: path)
    monkeypatch.setattr(bootstrap, "HFModelConfig", lambda **k: SimpleNamespace())
    monkeypatch.setattr(bootstrap, "omega_conf_to_dataclass", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(bootstrap, "prepared_generation_config", lambda *a: {})
    monkeypatch.setattr(bootstrap, "engine_contract", lambda *a: {})
    bootstrap.finalize_bootstrap(cfg)
    result = root.manifest(complete=True)
    assert ("source" in result["metadata"]["children_sha256"]) == enabled
    assert (root.root / "source").exists() == enabled


def test_final_export_verifies_public_bank_without_source(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file

    from verl.trainer.evaluation import checkpoint
    from verl.trainer.evaluation.artifacts import file_sha256

    bank = ArtifactStore.create(tmp_path / "initial", kind="generation", metadata={})
    bank.finish()
    refs = [dict(id=str(i), bank="public_initial", artifact=str(bank.root), valid_tokens=i) for i in range(6)]

    def score(trainer, selected, **kwargs):
        return [
            dict(id=r["id"], rkl_sum=0.25 * r["valid_tokens"], tokens=r["valid_tokens"], measure_id="m")
            for r in selected
        ], {}

    def save(destination, *args, **kwargs):
        from pathlib import Path

        hf = Path(destination) / "huggingface"
        hf.mkdir(parents=True)
        save_file({"weight": torch.ones(2)}, str(hf / "model.safetensors"))

    trainer = SimpleNamespace(
        global_steps=50,
        actor_optimizer_step=1600,
        actor_rollout_wg=SimpleNamespace(save_checkpoint=save, load_checkpoint=lambda *a, **k: None),
    )
    monkeypatch.setattr(checkpoint, "score_native_rows", score)
    checkpoint.final_model_export(trainer, refs, hidden_size=2, budget=None, destination=tmp_path / "final")
    audit = ArtifactStore(tmp_path / "final/export-audit").manifest(complete=True)["metadata"]
    assert audit["reference_bank"] == "public_initial" and audit["reference_ids"] == ["1", "2", "3", "4"]
    assert audit["reference_generation_bank"] == str(bank.root)
    assert audit["reference_generation_sha256"] == file_sha256(bank.manifest_path)
    assert audit["max_question_rkl_error"] == 0


def test_export_without_any_fixed_reference_fails_before_saving(tmp_path):
    from verl.trainer.evaluation.checkpoint import final_model_export

    with pytest.raises(ValueError, match="nonempty fixed reference"):
        final_model_export(None, [], hidden_size=2, budget=None, destination=tmp_path / "final")
    assert not (tmp_path / "final").exists()


def test_empty_final_boundary_still_finishes_run(tmp_path):
    coordinator = object.__new__(EvaluationCoordinator)
    coordinator.options = options(
        tmp_path,
        final_export=False,
        schedules={"banks": {"at": [0], "final": False}, "bench": {"at": [0], "final": False}},
    )
    coordinator.pending_outer = 50
    coordinator.completed = set()
    coordinator.pending_events = {}
    coordinator.store = ArtifactStore.create(tmp_path / "run", kind="test", metadata={})
    trainer = SimpleNamespace(total_training_steps=50, actor_optimizer_step=1600)
    assert coordinator.validate(trainer) == {}
    coordinator.on_readout(trainer, 50)
    assert coordinator.store.manifest(complete=True)["metadata"]["optimizer_steps"] == 1600


def test_disabled_history_does_not_emit_fake_zero_history_metrics():
    from verl.trainer.evaluation.metrics import readout_metrics

    metrics = readout_metrics([dict(id="p", bank="public_initial", rkl_sum=0.2, tokens=2)])
    assert not any(key.startswith("eval/rkl/history/") for key in metrics)
