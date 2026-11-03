# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Protocol preparation preserves weights and rejects undeclared tokenizer changes."""

import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import BPE

from examples.mirroropd.assets import stage_protocols as protocols


def definition(special="<eos>"):
    tokenizer = Tokenizer(BPE(vocab={"a": 0, "b": 1, "ab": 2}, merges=[("a", "b")]))
    tokenizer.add_special_tokens([special])
    return json.loads(tokenizer.to_str())


def test_only_explicit_token_labels_may_change():
    left, right = definition("<old>"), definition("<new>")
    assert protocols.verify_tokenizer_change(left, right, [3]) == [
        {"id": 3, "published": "<old>", "effective": "<new>"}
    ]
    with pytest.raises(ValueError, match="Undeclared"):
        protocols.verify_tokenizer_change(left, right, [])
    right["model"]["merges"] = []
    with pytest.raises(ValueError, match="segmentation"):
        protocols.verify_tokenizer_change(left, right, [3])


def test_old_and_new_bpe_serializations_have_the_same_meaning():
    left, right = definition(), definition()
    left["model"]["merges"] = ["a b"]
    left["model"].pop("ignore_merges", None)
    assert protocols.verify_tokenizer_change(left, right, []) == []


def test_preparation_preserves_originals_and_binds_an_immutable_view(monkeypatch, tmp_path):
    source, tokenizer, output = (tmp_path / key for key in ["source", "tokenizer", "output"])
    for path in [source, tokenizer]:
        path.mkdir()
        (path / "tokenizer.json").write_text(json.dumps(definition()))
        (path / "config.json").write_text(json.dumps({"eos_token_id": 3, "vocab_size": 4}))
        (path / "tokenizer_config.json").write_text(
            json.dumps({"bos_token": None, "eos_token": "<eos>", "pad_token": "<eos>", "chat_template": "canonical"})
        )
    (source / "model.safetensors").write_bytes(b"unchanged weight fixture")
    originals = {p.name: p.read_bytes() for p in source.iterdir()}
    targets = {"Olmo-3-1025-7B": source, "Olmo3-IVON-SFT-7B": tokenizer, "Olmo3-Base-BayesProtocol": output}
    monkeypatch.setattr(protocols, "registered_model", lambda name, **kwargs: targets[name])
    assert protocols.prepare("Olmo3-Base-BayesProtocol", root=tmp_path) == output
    assert {p.name: p.read_bytes() for p in source.iterdir()} == originals
    assert (output / "model.safetensors").resolve() == source / "model.safetensors"
    assert (output / "model.safetensors").read_bytes() == originals["model.safetensors"]
    assert json.loads((output / "config.json").read_text())["pad_token_id"] == 3
    assert protocols.prepare("Olmo3-Base-BayesProtocol", root=tmp_path) == output
    (output / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="metadata changed"):
        protocols.prepare("Olmo3-Base-BayesProtocol", root=tmp_path)
