# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Small CPU-only observations of current migration boundaries; no model loading."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from verl.models.transformers.full_vocab import install_full_vocab_adapter
from verl.trainer.distillation import full_vocab as kernels
from verl.trainer.distillation.hidden import FrozenTeacherHead
from verl.workers.utils.loss_context import LossContext
from verl.workers.utils.padding import prediction_positions


def rejected(fn):
    try:
        fn()
    except ValueError as error:
        return dict(rejected=True, reason=str(error))
    raise AssertionError('Expected this current interface to reject the request')


def main():
    torch.manual_seed(7)
    result = dict(device='cpu', torch=torch.__version__)
    result['llama_student_adapter'] = rejected(
        lambda: install_full_vocab_adapter(SimpleNamespace(config=SimpleNamespace(model_type='llama')))
    )
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, 'config.json').write_text(json.dumps(dict(model_type='llama')))
        result['llama_teacher_loader'] = rejected(lambda: FrozenTeacherHead(directory))
    hs, ws = torch.randn(2, 3, dtype=torch.float64), torch.randn(5, 3, dtype=torch.float64)
    ht, wt = torch.randn(2, 4, dtype=torch.float64), torch.randn(5, 4, dtype=torch.float64)
    calls, original = [], kernels.add_product

    def spy(accumulator, left, right):
        calls.append(list(accumulator.shape))
        return original(accumulator, left, right)

    kernels.add_product = spy
    try:
        with torch.no_grad():
            loss = kernels.full_reverse_kl(hs, ws, ht, wt, backend='precompute', token_chunk=2)
    finally:
        kernels.add_product = original
    assert not loss.requires_grad and calls == [[2, 3], [5, 3]]
    result['no_grad_precompute'] = dict(loss_requires_grad=loss.requires_grad, gradient_accumulator_shapes=calls)
    result['student_larger_support'] = rejected(
        lambda: kernels.full_reverse_kl(hs, torch.randn(6, 3, dtype=torch.float64), ht, wt)
    )
    wider = torch.randn(7, 4, dtype=torch.float64)
    actual = kernels.full_reverse_kl(hs, ws, ht, wider, backend='precompute', token_chunk=2)
    lp, lq = (hs @ ws.T).log_softmax(-1), (ht @ wider.T).log_softmax(-1)[:, :5]
    expected = (lp.exp() * (lp - lq)).sum()
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    result['native_teacher_extra'] = dict(matches_reference=True, absolute_error=float(abs(actual-expected)))
    valid = LossContext(dp_size=1, batch_num_tokens=6).normalize_token_sum(torch.tensor(12.0))
    result['normalization_example'] = dict(token_sum=12, actual_tokens=6, batch_size=2, horizon=4,
                                           valid_token_mean=float(valid), fixed_budget_mean=12/(2*4))
    masks = torch.nested.nested_tensor([torch.tensor([1, 1])], layout=torch.jagged)
    full = torch.nested.nested_tensor([torch.tensor([11, 12, 13, 21, 22])], layout=torch.jagged)
    truncated = torch.nested.nested_tensor([torch.tensor([11, 12, 13, 21])], layout=torch.jagged)
    result['prediction_positions'] = dict(full_input=prediction_positions(full, masks)[0][0].tolist(),
                                           prematurely_truncated=prediction_positions(truncated, masks)[0][0].tolist())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
