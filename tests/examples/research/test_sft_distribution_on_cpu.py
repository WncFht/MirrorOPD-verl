# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Verify sampling identities, complete-response budgets and exact merged-END gradients."""

import copy
import itertools
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.mirroropd.research.backends import hf_models
from examples.mirroropd.research.studies.fixed_transfer.sft_distribution import __main__ as cli
from examples.mirroropd.research.studies.fixed_transfer.sft_distribution import data, objectives, runtime


class Tiny(torch.nn.Module):
    def __init__(self, tied):
        super().__init__()
        self.embed = torch.nn.Embedding(23, 7)
        self.transform = torch.nn.Linear(7, 7, bias=False)
        self.head = torch.nn.Linear(7, 23, bias=False)
        if tied:
            self.head.weight = self.embed.weight

    def get_output_embeddings(self):
        return self.head


def tiny_hidden(model, rows, *, pad_id):
    assert len(rows) == 1
    row = rows[0]
    tokens = row["prompt_ids"] + row["response_ids"][:-1]
    hidden = model.transform(model.embed(torch.tensor(tokens))).tanh()
    # A causal prefix contribution makes prompt-mask mistakes visible in gradients.
    hidden = hidden + hidden.cumsum(0) / torch.arange(1, len(tokens) + 1)[:, None]
    return hidden[len(row["prompt_ids"]) - 1 :]


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("chunk", [1, 7, 64])
def test_end_merged_vjp_matches_autograd_with_accumulation(monkeypatch, tied, chunk):
    torch.manual_seed(77)
    actual = Tiny(tied)
    expected = copy.deepcopy(actual)
    monkeypatch.setattr(hf_models, "hidden", tiny_hidden)
    values = [dict(prompt_ids=[2, 3], content_ids=[4, 5, 6]), dict(prompt_ids=[7], content_ids=[8, 9] * 40)]
    observed, loss = 0.0, 0
    for v in values:
        observed += objectives.merged_ce_backward(actual, v, [20, 21], 37, chunk=chunk, support_size=22)
        states = tiny_hidden(
            expected, [dict(prompt_ids=v["prompt_ids"], response_ids=v["content_ids"] + [20])], pad_id=20
        )
        lp = objectives.log_probabilities(states, expected.head.weight, 22)
        body = lp[:-1].gather(1, torch.tensor(v["content_ids"])[:, None]).sum()
        loss -= (body + torch.logsumexp(lp[-1, [20, 21]], 0)) / 37
    loss.backward()
    assert observed == pytest.approx(float(loss.detach()), rel=2e-6)
    for a, b in zip(actual.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-6)
    # Padded output support receives no direct projection gradient.
    if not tied:
        assert torch.count_nonzero(actual.head.weight.grad[-1]) == 0


def test_sequence_score_and_rsr_match_dense_reference():
    torch.manual_seed(5)
    states, weight = torch.randn(5, 7), torch.randn(23, 7)
    content = [2, 4, 6, 8]
    lp = objectives.log_probabilities(states, weight, 22)
    result = objectives.merged_statistics(states, weight, content, [20, 21], chunk=2, support_size=22)
    chosen = lp[:-1].gather(1, torch.tensor(content)[:, None]).squeeze(1)
    assert result["sequence_logprob"] == pytest.approx(float(chosen.sum() + torch.logsumexp(lp[-1, [20, 21]], 0)))
    ranks = 1 + (lp[:-1] > chosen[:, None]).sum(-1)
    assert result["rsr"] == pytest.approx(float(ranks.sum() / -chosen.sum()))
    assert torch.isneginf(lp[:, -1]).all()


def candidates(n=8):
    return [
        dict(
            candidate_id=str(i),
            base=dict(sequence_logprob=-20.0 + i, mean_logprob=-10 + i, rsr=1 + i),
            expert=dict(sequence_logprob=-18.0 - i),
        )
        for i in range(n)
    ]


def test_ratio_extremes_and_calibration_are_stable():
    c = candidates()
    assert objectives.probabilities(c, "RATIO", 0) == objectives.probabilities(c, "RAW-C", 0)
    assert [i for i, p in enumerate(objectives.probabilities(c, "GRAPE", 1)) if p] == [4, 5, 6, 7]
    assert [i for i, p in enumerate(objectives.probabilities(c, "RSR", 1)) if p] == [0, 1, 2, 3]
    c[0]["base"]["sequence_logprob"] = -1e7
    p = objectives.probabilities(c, "RATIO", 1)
    assert math.fsum(p) == pytest.approx(1)
    assert p[0] == 0
    alpha, report = objectives.calibrate_alpha([c])
    assert alpha == min(report, key=lambda r: (abs(r["median_ess_fraction"] - 0.5), r["alpha"]))["alpha"]


def test_selection_table_shares_orders_and_is_reproducible():
    groups = [dict(id=str(i), candidates=candidates()) for i in range(7)]
    first = list(objectives.selection_table(groups, 42, 0.1))
    assert first == list(objectives.selection_table(groups, 42, 0.1))
    assert first != list(objectives.selection_table(groups, 43, 0.1))
    for epoch in range(1, 5):
        assert {v["question_id"] for v in first if v["epoch"] == epoch} == {str(i) for i in range(7)}


@pytest.mark.parametrize("length", [2048, 8192, 24576])
def test_full_response_budget_includes_real_stop(length):
    objectives.validate_content([2] * 4096, [3] * (length - 1), [20, 21])
    if length == 24576:
        with pytest.raises(ValueError, match="truncation"):
            objectives.validate_content([2], [3] * length, [20, 21])
    with pytest.raises(ValueError, match="internal"):
        objectives.validate_content([2], [3, 20, 4], [20, 21])


def test_near_duplicate_prefix_join_matches_exhaustive_graph():
    rng = random.Random(4)
    texts = [" ".join(f"w{v}" for v in rng.choices(range(100), k=50)) for _ in range(15)]
    texts += [t + " extra" for t in texts]
    texts += ["Ａ  b c", "A b   c"]
    actual = data.near_components(texts, 0.8)
    expected = list(range(len(texts)))

    def find(i):
        while expected[i] != i:
            i = expected[i]
        return i

    for i, j in itertools.combinations(range(len(texts)), 2):
        words = [data.text_key(texts[k]).casefold().split() for k in (i, j)]
        a, b = [set(zip(*(w[h:] for h in range(4)), strict=False)) or {tuple(w)} for w in words]
        if len(a & b) >= 0.8 * len(a | b):
            expected[find(i)] = find(j)
    for i, j in itertools.combinations(range(len(texts)), 2):
        assert (actual[i] == actual[j]) == (find(i) == find(j))


def test_guard_waits_on_ownership_and_never_takes_other_gpus(monkeypatch, tmp_path):
    calls = []

    def run(repo, gpus, command):
        calls.append(gpus)
        if len(calls) == 1:
            raise ValueError("GPU 0 allocation is already locked")
        return 0

    monkeypatch.setattr(cli, "contract", lambda root: {})
    monkeypatch.setattr(
        cli, "resource_guard", lambda: SimpleNamespace(parse_gpus=lambda g: list(map(int, g.split(","))), run=run)
    )
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    cli.wait_run(tmp_path, "0,5,6,7", 60)
    assert calls == ["0,5,6,7"] * 2
    with pytest.raises(ValueError, match="restricted"):
        cli.wait_run(tmp_path, "1,2,3,4", 60)


def test_pipeline_calibration_failure_prevents_bulk_generation(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "contract", lambda _: dict(expert_tp=2, seeds=[1, 2]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,5,6,7")
    monkeypatch.setattr(cli, "resource_guard", lambda: SimpleNamespace(verify_allocation=lambda *_: None))
    calls = []

    def invoke(root, stage, **kwargs):
        calls.append((stage, kwargs))
        if stage == "collect":
            raise ValueError("Calibration failed")

    monkeypatch.setattr(cli, "invoke", invoke)
    with pytest.raises(ValueError, match="Calibration"):
        cli.pipeline(tmp_path)
    assert not any(kwargs.get("split") == "train" for _, kwargs in calls)


def test_frozen_input_tampering_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "contract", lambda _: dict(seeds=[42]))
    data.write(tmp_path / "protocol.json", {})
    data.jsonl(tmp_path / "pool.jsonl", [])
    data.write(
        tmp_path / "frozen.json",
        dict(
            protocol_sha256=data.sha(tmp_path / "protocol.json"),
            files={"pool.jsonl": data.sha(tmp_path / "pool.jsonl")},
        ),
    )
    (tmp_path / "pool.jsonl").write_text("changed\n")
    with pytest.raises(ValueError, match="Frozen input"):
        runtime.load_training_inputs(tmp_path, 42)


def test_slot_seed_survives_sharding_and_resume():
    slots = [runtime.sample_seed("q", i, 42) for i in range(64)]
    assert len(set(slots)) == 64
    assert slots[16:32] == [runtime.sample_seed("q", i, 42) for i in range(16, 32)]


def test_recorded_density_marginalizes_both_stop_draws():
    values = [
        {4: SimpleNamespace(logprob=-0.3)},
        {5: SimpleNamespace(logprob=-0.4)},
        {20: SimpleNamespace(logprob=-1.0), 21: SimpleNamespace(logprob=-2.0)},
    ]
    body, end = runtime.sampled_path_probabilities([4, 5, 20], values, [20, 21])
    assert body == pytest.approx(-0.7)
    assert end == pytest.approx(math.log(math.exp(-1) + math.exp(-2)))
    assert runtime.sampled_path_probabilities([4, 5], values[:2], [20, 21]) == (None, None)
    assert runtime.sampled_path_probabilities([4, 5, 20], values[:2] + [{20: values[-1][20]}], [20, 21]) == (None, None)


def test_real_optimizer_resume_matches_uninterrupted_sft(monkeypatch, tmp_path):
    import numpy as np
    import transformers

    from examples.mirroropd.research.studies.fixed_transfer import teacher_construction_llm as old

    torch.manual_seed(5)
    initial = Tiny(True)
    groups = {
        str(i): dict(id=str(i), prompt_ids=[2, 3], candidates=[dict(candidate_id="c", content_ids=[4, 5, 6])])
        for i in range(2)
    }
    draws = [dict(epoch=e, question_id=str(i), choices={"RAW-C": "c"}) for e in range(1, 5) for i in range(2)]
    cfg = dict(
        base="initial",
        batch=1,
        learning_rate=3e-6,
        max_prompt=4096,
        max_response=24576,
        max_length=32768,
        end_ids=[20, 21],
        reference_length=8,
        token_chunk=2,
        support_size=22,
        clip=1.0,
    )
    monkeypatch.setattr(runtime, "load_training_inputs", lambda root, seed: (cfg, groups, draws))
    monkeypatch.setattr(hf_models, "hidden", tiny_hidden)
    monkeypatch.setattr(old, "train_mode", lambda model: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda _: None)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: None)

    def load(path, trainable):
        model = copy.deepcopy(initial)
        if str(path) != "initial":
            model.load_state_dict(torch.load(Path(path) / "weights.pt", weights_only=True))
        return model

    monkeypatch.setattr(old, "load_model", load)

    def checkpoint(model, tokenizer, optimizer, path, step):
        path.mkdir(parents=True)
        torch.save(model.state_dict(), path / "weights.pt")
        torch.save(
            dict(
                optimizer=optimizer.state_dict(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.get_rng_state(),
                python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),
            ),
            path / "training-state.pt",
        )

    monkeypatch.setattr(old, "checkpoint", checkpoint)
    for name in ("full", "resumed"):
        root = tmp_path / name
        data.write(root / "protocol.json", {})
        data.write(root / "frozen.json", {})
        data.write(root / "gpu-acceptance.json", dict(passed=True, protocol_sha256=data.sha(root / "protocol.json")))
    runtime.train(tmp_path / "full", "RAW-C", 42)
    real_append = old.append_log

    def interrupted_append(path, value):
        real_append(path, value)
        if value["step"] == 5:
            raise RuntimeError("Simulated interruption after a real optimizer step")

    monkeypatch.setattr(old, "append_log", interrupted_append)
    with pytest.raises(RuntimeError, match="Simulated"):
        runtime.train(tmp_path / "resumed", "RAW-C", 42)
    monkeypatch.setattr(old, "append_log", real_append)
    runtime.train(tmp_path / "resumed", "RAW-C", 42, tmp_path / "resumed/teachers/RAW-C-42/step0004")
    end = "teachers/RAW-C-42/step0008"
    expected = torch.load(tmp_path / "full" / end / "weights.pt", weights_only=True)
    actual = torch.load(tmp_path / "resumed" / end / "weights.pt", weights_only=True)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    metrics = list(data.records(tmp_path / "resumed/teachers/RAW-C-42/metrics.jsonl"))
    assert [v["step"] for v in metrics] == list(range(1, 9))
    assert list((tmp_path / "resumed/teachers/RAW-C-42").glob("metrics-before-resume-*.jsonl"))


def test_generation_quota_resume_and_sampler_density_collection(monkeypatch, tmp_path):
    import concurrent.futures

    import transformers

    cfg = dict(
        generation_shards=2,
        counts=dict(train=2),
        min_correct=8,
        initial_candidates=16,
        max_candidates=64,
        expert="unused",
        expert_tp=2,
        generation_seed=42,
        max_length=32768,
        max_response=24576,
        max_prompt=4096,
        end_ids=[20, 21],
        support_size=22,
    )
    monkeypatch.setattr(runtime, "contract", lambda _: cfg)
    data.write(tmp_path / "protocol.json", cfg)
    qs = [dict(id=str(i), prompt_ids=[2], expert_prompt_ids=[2, 3], ground_truth="1") for i in range(4)]
    data.jsonl(tmp_path / "prepared/train.jsonl", qs)
    loaded = []

    class FakeLLM:
        def __init__(self, **kwargs):
            assert kwargs["logprobs_mode"] == "processed_logprobs"
            loaded.append(True)

        def generate(self, prompts, params, use_tqdm):
            assert all(p.logprob_token_ids == [20, 21] for p in params)
            lp = [
                {4: SimpleNamespace(logprob=-0.3)},
                {5: SimpleNamespace(logprob=-0.4)},
                {20: SimpleNamespace(logprob=-1), 21: SimpleNamespace(logprob=-2)},
            ]
            return [
                SimpleNamespace(
                    outputs=[SimpleNamespace(token_ids=[4, 5, 20], stop_reason=20, finish_reason="stop", logprobs=lp)]
                )
                for _ in prompts
            ]

        def get_tokenizer(self):
            return SimpleNamespace(decode=lambda *a, **kw: "\\boxed{1}")

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **kw: SimpleNamespace(**kw))
    )
    monkeypatch.setattr(
        concurrent.futures, "ProcessPoolExecutor", lambda **kwargs: concurrent.futures.ThreadPoolExecutor(max_workers=2)
    )
    monkeypatch.setattr(runtime, "grade_candidate", lambda _: dict(correct=True, native_score=1, repeat8=0))
    for i in range(2):
        runtime.generate(tmp_path, "train", i, 2)
    assert len(loaded) == 2
    assert len(list((tmp_path / "generated/train").glob("*.json"))) == 2
    for i in range(2):
        runtime.generate(tmp_path, "train", i, 2)
    assert len(loaded) == 2

    def no_gpu(*args, **kwargs):
        raise AssertionError("Expert train scoring must reuse recorded density")

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", no_gpu)
    for i in range(2):
        runtime.score(tmp_path, "train", "expert", i, 2)
    for q in qs[:2]:
        generated = tmp_path / "generated/train" / f"{q['id']}.json"
        cs = data.read(generated)["candidates"]
        data.write(
            tmp_path / "scores/base/train" / f"{q['id']}.json",
            dict(
                generated_sha256=data.sha(generated),
                candidates={c["candidate_id"]: dict(sequence_logprob=-2, mean_logprob=-1, rsr=1) for c in cs},
            ),
        )
    runtime.collect_scores(tmp_path, "train")
    assert data.read(tmp_path / "quality-train.json")["covered"] == 2
    group = data.read(tmp_path / "scored/train/0.json")
    assert len(group["candidates"]) == 16
    assert group["candidates"][0]["expert"]["sequence_logprob"] == pytest.approx(
        -0.7 + math.log(math.exp(-1) + math.exp(-2))
    )


def test_acceptance_control_flow_and_dense_comparison_on_tiny_cpu_model(monkeypatch, tmp_path):
    from examples.mirroropd.research.studies.fixed_transfer import teacher_construction_llm as old

    model = Tiny(True)
    model.embed = torch.nn.Embedding(256, 7)
    model.head = torch.nn.Linear(7, 256, bias=False)
    model.head.weight = model.embed.weight
    model.get_input_embeddings = lambda: model.embed
    cfg = dict(
        base="unused",
        seeds=[42],
        learning_rate=3e-6,
        max_prompt=4096,
        max_response=24576,
        max_length=32768,
        end_ids=[20, 21],
        support_size=256,
        reference_length=8192,
        token_chunk=64,
        clip=1.0,
    )
    monkeypatch.setattr(runtime, "contract", lambda _: cfg)
    monkeypatch.setattr(old, "load_model", lambda *args: model)
    monkeypatch.setattr(old, "train_mode", lambda _: None)
    monkeypatch.setattr(hf_models, "hidden", tiny_hidden)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    data.write(tmp_path / "protocol.json", cfg)
    data.jsonl(tmp_path / "prepared/calibration.jsonl", [dict(prompt_ids=[2, 3])])
    runtime.acceptance(tmp_path)
    result = data.read(tmp_path / "gpu-acceptance.json")
    assert result["passed"] and result["tied_embeddings"]
    assert [r["length"] for r in result["checks"]] == [2048, 8192, 24576]
    assert all(r["prompt_tokens"] == 4096 for r in result["checks"])
    assert result["checks"][0]["dense_comparison"]["relative_gradient_l2"] < 1e-5
