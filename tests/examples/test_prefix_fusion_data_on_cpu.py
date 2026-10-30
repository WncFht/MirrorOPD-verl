# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Scientific evaluation questions never enter the dynamic rollout training pool."""

from examples.mirroropd.research.studies.prefix_fusion.data import is_holdout_group, select_training
from examples.mirroropd.research.studies.prefix_fusion.protocol import ARMS, PROTOCOL


def row(text):
    return {"prompt": [{"role": "user", "content": text}]}


def test_exclude_both_fixed_states_and_task_questions_without_mutating_source():
    rows = [
        row("training one"),
        row(" heldout   question "),
        row("task question"),
        row("training   one"),
        row("training two"),
    ]
    heldout = [{"messages": row("heldout question")["prompt"]}]
    tasks = [{"messages": row("task   question")["prompt"]}]
    selected, removed = select_training(rows, heldout, tasks)
    assert selected == [0, 4]
    assert removed == {"heldout_or_task_overlap": 2, "duplicate_training_question": 1}
    assert len(rows) == 5 and rows[1]["prompt"][0]["content"] == " heldout   question "


def test_training_group_m_is_not_a_math_holdout_group():
    assert not is_holdout_group("M")
    assert is_holdout_group("M01") and is_holdout_group("M12")
    assert is_holdout_group("Code01") and is_holdout_group("IF04")
    assert not is_holdout_group("A") and not is_holdout_group("Math")


def test_six_dynamic_sources_share_readouts_and_budget():
    assert set(ARMS) == {"S0", "S", "PM", "LM", "TM", "T"}
    assert ARMS["S"]["mode"] == "trajectory_mixture" and ARMS["S"]["teacher_weight"] == 0
    assert ARMS["PM"]["fusion_space"] == "probability"
    assert ARMS["LM"]["fusion_space"] == "logit"
    assert ARMS["TM"]["mode"] == "trajectory_mixture"
    assert PROTOCOL["training"]["outers"] * PROTOCOL["training"]["reuse"] == 640
    assert PROTOCOL["fixed_risk"]["outers"] == [0, 5, 10, 15, 20, 25, 30, 35, 40]
    assert PROTOCOL["accuracy"]["models"].startswith("plain_student")
    assert PROTOCOL["checkpoints"]["persistent_weight_outers"] == [20, 40]


def test_bank_rng_is_per_question_and_short_view_does_not_regenerate():
    from examples.mirroropd.research.studies.prefix_fusion.banks import bank_request

    math_row = {"id": 321, "domain": "math"}
    code_row = {"id": 512, "domain": "code"}
    before = bank_request(math_row, "S0")
    bank_request(code_row, "P")
    assert bank_request(math_row, "S0") == before
    assert before["seed"] != bank_request(math_row, "J")["seed"]
    assert before["cap"] == 7168 and bank_request(code_row, "S0")["cap"] == 1024


def test_seeded_stateful_loader_matches_frozen_order():
    import torch
    from torchdata.stateful_dataloader import StatefulDataLoader
    from torchdata.stateful_dataloader.sampler import RandomSampler

    dataset = list(range(512))
    expected = list(RandomSampler(dataset, generator=torch.Generator().manual_seed(42)))[:256]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(999)
        loader = StatefulDataLoader(
            dataset,
            batch_size=64,
            num_workers=0,
            sampler=RandomSampler(dataset, generator=torch.Generator().manual_seed(42)),
        )
        actual = []
        for i, batch in enumerate(loader):
            actual.extend(batch.tolist())
            if i == 3:
                break
    assert actual == expected
