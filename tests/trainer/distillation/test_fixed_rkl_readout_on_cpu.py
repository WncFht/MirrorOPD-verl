# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent RKL oracle and history sampling contracts."""

from collections import Counter

import pytest
import torch

from verl.trainer.distillation.evaluation import sample_sums, score_rkl_rows
from verl.trainer.distillation.numerics import kl_rows, kl_values
from verl.trainer.distillation.readout import balanced_history


@pytest.mark.parametrize("teacher_vocab", [7, 11])
def test_readout_matches_independent_float64_oracle(teacher_vocab):
    g = torch.Generator().manual_seed(17)
    hs = torch.randn(9, 3, generator=g, dtype=torch.float64, requires_grad=True)
    ws = torch.randn(7, 3, generator=g, dtype=torch.float64, requires_grad=True)
    ht = torch.randn(9, 5, generator=g, dtype=torch.float64)
    wt = torch.randn(teacher_vocab, 5, generator=g, dtype=torch.float64)
    before = hs.detach().clone(), ws.detach().clone()
    lp = (hs @ ws.T).log_softmax(-1)
    lq = (ht @ wt.T).log_softmax(-1)
    expected = (lp.exp() * (lp - lq[:, :7])).sum(-1)
    for chunk in [1, 4, 32]:
        actual = score_rkl_rows(hs, ws, ht, wt, token_chunk=chunk)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        assert not actual.requires_grad
    assert hs.grad is None and ws.grad is None
    assert torch.equal(hs, before[0]) and torch.equal(ws, before[1])


def test_copy_and_near_copy_and_value_only_regression():
    g = torch.Generator().manual_seed(31)
    z = torch.randn(11, 19, generator=g, dtype=torch.float64)
    torch.testing.assert_close(kl_values(z, z), torch.zeros(11, dtype=torch.float64), atol=1e-14, rtol=0)
    teacher = z + 1e-5 * torch.randn(z.shape, generator=g, dtype=z.dtype)
    lp, lq = z.log_softmax(-1), teacher.log_softmax(-1)
    expected = (lp.exp() * (lp - lq)).sum(-1)
    torch.testing.assert_close(kl_values(z, teacher), expected, atol=2e-15, rtol=1e-5)
    torch.testing.assert_close(kl_values(z, teacher), kl_rows(z, teacher)[0], atol=0, rtol=0)


def test_zero_mask_and_output_row_alignment():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sample_sums(values, [2, 0, 3])
    assert result["rkl_sum"].tolist() == [[3.0], [0.0], [12.0]]
    assert result["rkl_count"].tolist() == [[2], [0], [3]]
    assert sample_sums(values[:0], [])["rkl_sum"].shape == (0, 1)
    with pytest.raises(ValueError):
        sample_sums(values, [2, 2])


def test_fifty_batch_history_is_balanced_nested_and_reproducible():
    batches, previous = {}, {}
    for m in range(1, 51):
        batches[m] = [dict(source_id=f"{m}:{i}") for i in range(128)]
        rows = balanced_history(batches, 128, 42)
        assert rows == balanced_history(batches, 128, 42)
        assert len(rows) == len({r["source_id"] for r in rows}) == 128
        counts = Counter(r["history_batch"] for r in rows)
        assert set(counts) == set(range(1, m + 1))
        assert max(counts.values()) - min(counts.values()) <= 1
        current = {b: {r["source_id"] for r in rows if r["history_batch"] == b} for b in counts}
        for b in previous:
            assert current[b] <= previous[b]
        previous = current
    assert sorted(counts.values()) == [2] * 22 + [3] * 28
    with pytest.raises(ValueError):
        balanced_history(batches, 49, 42)


def test_engine_readout_keeps_sample_order_and_masked_positions():
    from types import SimpleNamespace

    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    from verl.workers.utils.padding import prediction_positions

    def nested(rows, dtype):
        return torch.nested.nested_tensor([torch.tensor(r, dtype=dtype) for r in rows], layout=torch.jagged)

    data = TensorDict(
        {
            "input_ids": nested([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], torch.int64),
            "responses": nested([[3, 4, 5], [9, 10]], torch.int64),
            "loss_mask": nested([[1, 0, 1], [1, 1]], torch.int64),
        },
        batch_size=[2],
    )
    positions, packed = prediction_positions(data["input_ids"], data["loss_mask"])
    data["teacher_positions"] = torch.nested.as_nested_tensor(positions, layout=torch.jagged)
    g = torch.Generator().manual_seed(71)
    hidden = torch.randn(1, 10, 3, generator=g, dtype=torch.float64)
    ws = torch.randn(7, 3, generator=g, dtype=torch.float64)
    ht = torch.randn(4, 5, generator=g, dtype=torch.float64)
    wt = torch.randn(9, 5, generator=g, dtype=torch.float64)
    data["teacher_hidden"] = torch.nested.as_nested_tensor(list(ht.split([2, 2])), layout=torch.jagged)
    tu.assign_non_tensor(data, full_vocab_score=True, temperature=1.0)

    class Head:
        def __call__(self, h, *, full_vocab_callback):
            return full_vocab_callback(h, ws)

    class Model:
        def __call__(self, **kwargs):
            return kwargs["full_vocab_callback"](hidden, Head())

    engine = FSDPEngineWithLMHead.__new__(FSDPEngineWithLMHead)
    engine.model_config = SimpleNamespace(enable_full_vocab=True)
    engine.full_vocab_teacher_head = SimpleNamespace(get=lambda device, dtype: wt)
    engine.full_vocab_config = SimpleNamespace(token_chunk_size=2)
    engine.full_vocab_spec = SimpleNamespace()
    engine.module = Model()
    result = engine._full_vocab_output(data, {}, teacher=False)
    lp = (hidden[0, packed] @ ws.T).log_softmax(-1)
    lq = (ht @ wt.T).log_softmax(-1)
    expected = (lp.exp() * (lp - lq[:, :7])).sum(-1)
    torch.testing.assert_close(result["rkl_sum"][:, 0], torch.stack([expected[:2].sum(), expected[2:].sum()]))
    assert result["rkl_count"].tolist() == [[2], [2]]


def test_capture_rows_accepts_tq_linked_list_metadata(monkeypatch):
    from types import SimpleNamespace

    from tensordict import TensorDict
    from tensordict.utils import LinkedList

    from verl.trainer.distillation import readout

    def nested(rows):
        return torch.nested.nested_tensor([torch.tensor(r) for r in rows], layout=torch.jagged)

    stored = dict(
        prompts=nested([[1, 2], [3]]),
        responses=nested([[4, 5], [6, 7, 8]]),
        response_mask=nested([[1, 1], [1, 1, 0]]),
        extra_info=LinkedList([{"index": 21}, {"index": 22}], td=TensorDict({}, batch_size=[2])),
    )
    monkeypatch.setattr(readout.tq, "kv_batch_get", lambda **kwargs: stored)
    batch = SimpleNamespace(keys=["a", "b"], partition_id="val", tags=[{}, {}])
    rows = readout.capture_rows(batch, 0)
    assert [r["source_id"] for r in rows] == ["21", "22"]
    assert rows[1]["response_ids"] == [6, 7, 8]
    assert rows[1]["response_mask"] == [1, 1, 0]
