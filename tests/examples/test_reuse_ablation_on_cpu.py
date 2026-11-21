# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Ablations change reuse and clocks while preserving optimizer and evaluation identity."""

import copy

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from examples.mirroropd.tools.reuse_ablation import ARMS, audit_prompt_pool, configure, prepare_training_pool


def test_all_arms_share_inputs_optimizer_reporting_banks_and_update_budget(tmp_path):
    reference = dict(
        data=dict(seed=42, train_files=["/canonical/train.parquet"], train_batch_size=64),
        trainer=dict(
            n_gpus_per_node=4,
            evaluation=dict(
                bootstrap="/canonical/bootstrap",
                datasets=[dict(path="/canonical/math.parquet")],
                protocols=dict(public=dict(seed=42)),
            ),
        ),
        actor_rollout_ref=dict(
            model=dict(path="/canonical/student"),
            actor=dict(
                optim=dict(optimizer="MuonAdamW", lr=3e-6, seed=42, override_optimizer_config=dict(aux_lr=1e-6)),
                checkpoint={},
            ),
            rollout=dict(seed=42, response_length=7168),
        ),
        ray_kwargs=dict(ray_init=dict(runtime_env=dict(env_vars=dict(PYTHONPATH="/old/source")))),
    )
    frozen = copy.deepcopy(reference)
    outputs = set()
    for arm, reuse in ARMS.items():
        config = configure(
            reference,
            name=arm,
            arm=arm,
            root=tmp_path,
            source=tmp_path / "source",
            original_source="/old/source",
            updates=640,
            every=160,
            seed=42,
        )
        assert config["data"] == reference["data"]
        optimizer = config["actor_rollout_ref"]["actor"]["optim"]
        assert optimizer["optimizer"] == "AdamW" and optimizer["optimizer_impl"] == "torch.optim"
        assert optimizer["lr"] == 1e-6 and optimizer["weight_decay"] == 0.01
        assert optimizer["betas"] == [0.9, 0.999]
        assert optimizer["override_optimizer_config"] == dict(eps=1e-8)
        assert config["actor_rollout_ref"]["actor"]["ppo_epochs"] == reuse
        evaluation = config["trainer"]["evaluation"]
        for key in ("bootstrap", "datasets", "protocols"):
            assert evaluation[key] == reference["trainer"]["evaluation"][key]
        assert config["trainer"]["update_budget"] == dict(total=640, eval_every=160, save_every=160)
        assert config["trainer"]["reuse_control"]["mode"] == ("adaptive" if arm == "adaptive" else "fixed")
        assert evaluation["export_outers"] == [] and evaluation["final_export"]
        outputs.add(evaluation["output"])
        assert config["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]["PYTHONPATH"] == str(tmp_path / "source")
    assert list(ARMS) == ["adaptive", "fixed16", "fixed8", "fixed4", "online"]
    assert len(outputs) == 5 and reference == frozen


def test_recycled_pool_audit_checks_every_question_not_just_initial_training_prefix(tmp_path):
    train, heldout = tmp_path / "train.parquet", tmp_path / "math.parquet"
    prompts = [[dict(role="user", content=f"Problem {i}")] for i in range(70)]
    pq.write_table(pa.Table.from_pylist([dict(prompt=p) for p in prompts]), train)
    pq.write_table(pa.Table.from_pylist([dict(prompt=[dict(role="user", content="Held out")])]), heldout)
    cfg = dict(
        data=dict(train_files=[str(train)], train_batch_size=64),
        trainer=dict(evaluation=dict(datasets=[dict(path=str(heldout))])),
    )
    assert audit_prompt_pool(cfg) == dict(rows=70, unique_texts=70, heldout_text_overlap=0)
    pq.write_table(pa.Table.from_pylist([dict(prompt=[dict(role="user", content="  PROBLEM   69 ")])]), heldout)
    with pytest.raises(ValueError, match="overlaps"):
        audit_prompt_pool(cfg)
    original = train.read_bytes()
    study = tmp_path / "study"
    study.mkdir()
    prepare_training_pool(cfg, study)
    assert audit_prompt_pool(cfg) == dict(rows=69, unique_texts=69, heldout_text_overlap=0)
    assert train.read_bytes() == original
