# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch

from verl.experimental.rollout_policy.vllm_support import TokenizerSupport


def test_tokenizer_support_masks_unmapped_head_rows_and_padding(monkeypatch):
    tokenizer = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 2, "end": 4})
    monkeypatch.setattr(
        "verl.experimental.rollout_policy.vllm_support.AutoTokenizer.from_pretrained", lambda *a, **kw: tokenizer
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            tokenizer="local", model="local", trust_remote_code=False, get_vocab_size=lambda: 8
        )
    )
    processor = TokenizerSupport(config, torch.device("cpu"), False)
    logits = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    actual = processor.apply(logits)
    assert actual.data_ptr() == logits.data_ptr()
    assert actual.argmax(-1).tolist() == [4, 4]
    assert torch.isneginf(actual[:, [1, 3, 5, 6, 7, 8, 9]]).all()
    assert torch.isfinite(actual[:, [0, 2, 4]]).all()
    assert not processor.is_argmax_invariant()


def test_processor_resolves_through_vllm_loader():
    from vllm.v1.sample.logits_processor import _load_logitsprocs_by_fqcns

    loaded = _load_logitsprocs_by_fqcns(("verl.experimental.rollout_policy.vllm_support:TokenizerSupport",))
    assert TokenizerSupport in loaded
