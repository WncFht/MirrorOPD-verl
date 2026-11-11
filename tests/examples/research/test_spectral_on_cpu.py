# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from examples.mirroropd.research.studies.spectral.decomposition import (
    energy,
    frame_components,
    head_terms,
    random_controls,
    top_fraction,
    weight_summary,
)
from examples.mirroropd.research.studies.spectral.residuals import (
    ResidualBank,
    analyze,
    fisher_residual,
    fit_right_basis,
    score_bands,
)
from examples.mirroropd.research.studies.spectral.weights import TensorCheckpoint, reconstruct


def test_frame_components_are_orthogonal_and_null_is_rectangular():
    g = torch.Generator().manual_seed(4)
    base = torch.randn(7, 4, generator=g, dtype=torch.float64)
    delta = torch.randn(7, 4, generator=g, dtype=torch.float64)
    parts = frame_components(base, delta)
    torch.testing.assert_close(sum(parts.values()), delta)
    for left, right in (("diag", "off"), ("diag", "null"), ("off", "null")):
        assert abs(float((parts[left] * parts[right]).sum())) < 1e-12
    torch.testing.assert_close(sum(energy(x) for x in parts.values()), energy(delta))
    assert energy(parts["null"]) > 0.1
    square = frame_components(base[:4], delta[:4])
    assert energy(square["null"]) < 1e-24


def test_iso_scalars_and_rank_are_calibrated():
    base = torch.diag(torch.tensor([8.0, 4.0, 1.0], dtype=torch.float64))
    delta = torch.diag(torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64))
    values = weight_summary(base, base + delta)
    assert values["kappa_spec"] == pytest.approx(3.0)
    assert values["rho_sigma"] == pytest.approx(1.0)
    assert weight_summary(base, base + 2 * delta)["kappa_spec"] == pytest.approx(values["kappa_spec"])
    assert weight_summary(base, base)["rho_sigma"] is None
    assert weight_summary(base, base)["stable_rank"] is None
    assert weight_summary(base, base)["numerical_rank"] == 0


def test_top_fraction_norm_restore_and_null_controls():
    delta = torch.diag(torch.tensor([4.0, 2.0, 1.0, 0.5], dtype=torch.float64))
    truncated = top_fraction(delta, 0.01)
    assert torch.linalg.matrix_rank(truncated) == 1
    assert energy(truncated) == 16
    torch.testing.assert_close(energy(top_fraction(delta, 0.01, restore_norm=True)), energy(delta))
    controls = random_controls(delta, seed=3)
    for control in controls.values():
        torch.testing.assert_close(energy(control), energy(delta))
    torch.testing.assert_close(torch.linalg.svdvals(controls["random_frames"]), torch.linalg.svdvals(delta))


def make_checkpoint(path, matrices, *, sharded=False):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(dict(tie_word_embeddings=True)))
    if sharded:
        index = {}
        for i, (key, value) in enumerate(matrices.items()):
            filename = f"part{i}.safetensors"
            save_file({key: value}, path / filename)
            index[key] = filename
        (path / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map=index)))
    else:
        save_file({k: v.clone() for k, v in matrices.items()}, path / "model.safetensors")
    return path


def test_checkpoint_reconstruction_preserves_sources_and_untouched_aliases(tmp_path):
    key = "model.layers.0.mlp.up_proj.weight"
    b = torch.eye(4)
    delta = torch.diag(torch.tensor([4.0, 2.0, 1.0, 0.5]))
    base = make_checkpoint(tmp_path / "base", {key: b, "model.embed_tokens.weight": b})
    target = make_checkpoint(tmp_path / "target", {key: b + delta, "model.embed_tokens.weight": b + 3}, sharded=True)
    output = tmp_path / "top"
    reconstruct(base, target, output, mode="top", untouched="base")
    result = TensorCheckpoint(output)
    torch.testing.assert_close(result.get(key), b + torch.diag(torch.tensor([4.0, 0.0, 0.0, 0.0])))
    torch.testing.assert_close(result.get("model.embed_tokens.weight"), b)
    torch.testing.assert_close(TensorCheckpoint(target).get(key), b + delta)
    assert json.loads((output / "spectral-intervention.json").read_text())["complete"]
    with pytest.raises(FileExistsError):
        reconstruct(base, target, output, mode="top")
    randomized = tmp_path / "randomized"
    reconstruct(base, target, randomized, mode="top", control="random_frames", seed=3)
    torch.testing.assert_close(
        torch.linalg.svdvals(TensorCheckpoint(randomized).get(key) - b),
        torch.tensor([4.0, 0.0, 0.0, 0.0]),
        atol=1e-6,
        rtol=1e-5,
    )
    deleted = tmp_path / "deleted"
    reconstruct(base, target, deleted, mode="delete-diag")
    torch.testing.assert_close(TensorCheckpoint(deleted).get(key), b)
    patched = tmp_path / "patched"
    reconstruct(base, base, patched, mode="patch-diag", anchor=target)
    torch.testing.assert_close(TensorCheckpoint(patched).get(key), b + delta)
    with pytest.raises(ValueError, match="tied-weight"):
        reconstruct(base, target, tmp_path / "bad", mode="top", pattern="embed_tokens")


def make_caches(tmp_path, *, alpha=0.5):
    generator = torch.Generator().manual_seed(8)
    base = torch.randn(12, 7, generator=generator, dtype=torch.float64).log_softmax(-1)
    teacher = (base + 0.5 * torch.randn(12, 7, generator=generator, dtype=torch.float64)).log_softmax(-1)
    opd = ((1 - alpha) * base + alpha * teacher).log_softmax(-1)
    positions = [dict(row_id=str(i // 3), split="train" if i < 6 else "heldout", position=i % 3) for i in range(12)]
    paths = []
    for name, logs in (("b", base), ("t", teacher), ("o", opd)):
        path = tmp_path / name
        path.mkdir()
        np.save(path / "logprobs.npy", logs.float().numpy())
        metadata = dict(positions=positions, vocabulary="same", bank="bank", shape=[12, 7], complete=True)
        (path / "metadata.json").write_text(json.dumps(metadata))
        paths.append(path)
    return paths


def test_fisher_residual_is_gauge_invariant():
    b = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64).log_softmax(-1)
    t = torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float64).log_softmax(-1)
    torch.testing.assert_close(fisher_residual(b, t), fisher_residual(b, t + 9))
    assert abs(float((b.exp().sqrt() * fisher_residual(b, t)).sum())) < 1e-14


def test_calibration_svd_matches_dense_and_absorption_is_signed(tmp_path):
    bank = ResidualBank(*make_caches(tmp_path, alpha=-0.5))
    cal = bank.indices("train")
    basis, diagnostics = fit_right_basis(bank, cal, rank=6, oversample=0, block=2)
    dense = torch.cat([rt for _, rt, _, _ in bank.blocks(cal, block=3)])
    torch.testing.assert_close(
        torch.tensor(diagnostics["singular_values"], dtype=torch.float64), torch.linalg.svdvals(dense)
    )
    scores = score_bands(bank, bank.indices("heldout"), basis, block=2)
    for row in scores:
        for band in row["bands"].values():
            if band["teacher_energy"] > 1e-12:
                assert band["absorption"] == pytest.approx(-0.5, abs=2e-6)
                assert band["missing"] == pytest.approx(2.25, abs=3e-6)
        parts = [v for k, v in row["bands"].items() if k != "all"]
        assert sum(v["teacher_energy"] for v in parts) == pytest.approx(row["bands"]["all"]["teacher_energy"])
    # Changing only heldout responses cannot rotate the calibration basis.
    with torch.no_grad():
        data = np.load(tmp_path / "t" / "logprobs.npy")
        data[6:] = np.roll(data[6:], 1, axis=1)
        np.save(tmp_path / "t" / "logprobs.npy", data)
    updated = ResidualBank(tmp_path / "b", tmp_path / "t", tmp_path / "o")
    other, _ = fit_right_basis(updated, cal, rank=6, oversample=0, block=2)
    torch.testing.assert_close(other @ other.T, basis @ basis.T)


def test_output_pipeline_and_support_mismatch(tmp_path):
    paths = make_caches(tmp_path)
    analyze(*paths, tmp_path / "result", rank=3, block=2)
    result = json.loads((tmp_path / "result" / "scores.json").read_text())
    assert len(result["evaluation"]) == len(result["random_evaluation"]) == 2
    assert result["diagnostics"]["relative_frobenius_error"] >= 0
    metadata = json.loads((paths[1] / "metadata.json").read_text())
    metadata["vocabulary"] = "wrong"
    (paths[1] / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        ResidualBank(*paths)


def test_exact_head_identity_and_centering():
    from examples.mirroropd.research.studies.spectral.representations import head_band_summary

    torch.manual_seed(7)
    h0, ht, ho = [torch.randn(5, 3, dtype=torch.float64) for _ in range(3)]
    w0, wt = [torch.randn(7, 3, dtype=torch.float64) for _ in range(2)]
    terms = head_terms(h0, ht, w0, wt)
    torch.testing.assert_close(sum(terms.values()), ht @ wt.T - h0 @ w0.T)
    result = head_band_summary(h0, ht, ho, w0, wt, rank=1)
    assert result["identity_error"] < 1e-12
    assert sum(x["teacher_energy"] for x in result["bands"].values()) == pytest.approx(float(energy(ht - h0)))


def test_extraction_prediction_offsets_hooks_and_complete_marker(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from examples.mirroropd.research.studies.spectral.extract import extract

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(7, 3)
            self.layers = torch.nn.ModuleList([torch.nn.Linear(3, 3)])
            self.norm = torch.nn.LayerNorm(3)

        def forward(self, input_ids, use_cache=False):
            value = self.embedding(input_ids)
            for layer in self.layers:
                value = layer(value)
            return SimpleNamespace(last_hidden_state=self.norm(value))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Decoder()
            self.lm_head = torch.nn.Linear(3, 7, bias=False)

        def get_output_embeddings(self):
            return self.lm_head

    model = Model().eval()
    path = make_checkpoint(tmp_path / "model", {"weight": torch.eye(3)})
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps([dict(prompt_ids=[1, 2], response_ids=[3, 4, 5], split="train", row_id="q")]))
    monkeypatch.setattr("examples.mirroropd.research.studies.spectral.extract.load_model", lambda *a, **k: model)
    import sys

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *a, **k: SimpleNamespace(get_vocab=lambda: {str(i): i for i in range(7)})
            )
        ),
    )
    output = tmp_path / "cache"
    extract(path, bank, output, device="cpu", positions=2, chunk=1, layers=[0])
    with torch.no_grad():
        expected = model.model(torch.tensor([[1, 2, 3, 4]])).last_hidden_state[0, [1, 3]]
        logs = model.lm_head(expected).log_softmax(-1)
    torch.testing.assert_close(torch.from_numpy(np.load(output / "hidden-final.npy")), expected)
    torch.testing.assert_close(torch.from_numpy(np.load(output / "logprobs.npy")), logs)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["complete"]
    assert [p["position"] for p in metadata["positions"]] == [0, 2]
    assert np.load(output / "hidden-layer-0.npy").shape == (2, 3)
    with pytest.raises(FileExistsError):
        extract(path, bank, output, device="cpu")


def test_teacher_extra_support_is_preserved_in_native_rkl(tmp_path):
    paths = make_caches(tmp_path)
    original = np.load(paths[1] / "logprobs.npy")
    extended = torch.cat([torch.from_numpy(original), torch.zeros(12, 2)], dim=1).log_softmax(-1).numpy()
    np.save(paths[1] / "logprobs.npy", extended)
    meta = json.loads((paths[1] / "metadata.json").read_text())
    meta["shape"] = [12, 9]
    (paths[1] / "metadata.json").write_text(json.dumps(meta))
    bank = ResidualBank(*paths)
    basis, _ = fit_right_basis(bank, bank.indices("train"), rank=3)
    scores = score_bands(bank, bank.indices("heldout"), basis)
    b = torch.from_numpy(np.load(paths[0] / "logprobs.npy")).double().log_softmax(-1)
    t = torch.from_numpy(extended).double().log_softmax(-1)
    expected = (b.exp() * (b - t[:, :7])).sum(-1)
    assert scores[0]["native_base_rkl"] == pytest.approx(float(expected[6:9].mean()))


def test_fisher_full_residual_reconstructs_teacher():
    from examples.mirroropd.research.studies.spectral.residuals import residual_distribution

    b = torch.randn(4, 8, dtype=torch.float64).log_softmax(-1)
    t = torch.randn(4, 8, dtype=torch.float64).log_softmax(-1)
    torch.testing.assert_close(residual_distribution(b, fisher_residual(b, t)), t)


@pytest.mark.parametrize("teacher_execution", ["native", "matched"])
def test_native_score_matches_dense_rkl_with_extra_teacher_support(tmp_path, monkeypatch, teacher_execution):
    # Initialize production imports before replacing the tokenizer boundary.
    import importlib
    import sys
    from types import SimpleNamespace

    from examples.mirroropd.research.studies.spectral.scoring import score

    importlib.import_module("verl.trainer.distillation.reverse_vocab")

    generator = torch.Generator().manual_seed(12)
    hs, ht = [torch.randn(3, 2, generator=generator) for _ in range(2)]
    ws = torch.randn(5, 2, generator=generator)
    wt = torch.randn(7, 2, generator=generator)
    model = SimpleNamespace(lm_head=SimpleNamespace(weight=ws), requires_grad_=lambda _: None)
    checkpoint = make_checkpoint(tmp_path / "model", {"weight": ws})
    records = tmp_path / "records.json"
    records.write_text(json.dumps([dict(prompt_ids=[1], response_ids=[2, 3, 4], split="test", row_id="q")]))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *a, **k: SimpleNamespace(get_vocab=lambda: {str(i): i for i in range(5)})
            )
        ),
    )
    execution_policies = []

    def cached_teacher(*args, student=False, **kwargs):
        execution_policies.append(student)
        return dict(hidden={"q": ht}, head=wt, identity={})

    monkeypatch.setattr("examples.mirroropd.research.studies.spectral.scoring.teacher_cache", cached_teacher)
    monkeypatch.setattr("examples.mirroropd.research.studies.spectral.scoring.load_model", lambda *a, **k: model)
    monkeypatch.setattr("examples.mirroropd.research.studies.spectral.scoring.response_hidden", lambda *a: hs)
    output = tmp_path / "risk.json"
    score(
        checkpoint,
        checkpoint,
        records,
        output,
        cache=tmp_path / "cache",
        device="cpu",
        cap=4,
        token_chunk=2,
        vocab_chunk=3,
        teacher_execution=teacher_execution,
    )
    assert execution_policies == [teacher_execution == "matched"]
    assert json.loads(output.read_text())["teacher_execution"] == teacher_execution
    lp = (hs @ ws.T).double().log_softmax(-1)
    lq = (ht @ wt.T).double().log_softmax(-1)[:, :5]
    expected = (lp.exp() * (lp - lq)).sum() / 4
    assert json.loads(output.read_text())["rows"][0]["loss"] == pytest.approx(float(expected), abs=1e-6)


def test_tiny_hf_no_edit_pipeline_preserves_ties_and_native_risk(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    from examples.mirroropd.research.studies.spectral.extract import extract
    from examples.mirroropd.research.studies.spectral.representations import analyze as hidden_analysis
    from examples.mirroropd.research.studies.spectral.representations import analyze_head
    from examples.mirroropd.research.studies.spectral.scoring import score

    path = tmp_path / "model"
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=8,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            tie_word_embeddings=True,
        )
    )
    model.save_pretrained(path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(8)}, unk_token="0")), unk_token="0"
    )
    tokenizer.save_pretrained(path)
    records = tmp_path / "bank.json"
    records.write_text(
        json.dumps(
            [
                dict(row_id=str(i), prompt_ids=[1, 2 + i], response_ids=[3, 4, 5], split=split)
                for i, split in enumerate(("train", "heldout"))
            ]
        )
    )
    cache = tmp_path / "extracted"
    extract(path, records, cache, positions=2, layers=[0], device="cpu")
    exported = tmp_path / "copy"
    reconstruct(path, path, exported, mode="top", fraction=1)
    restored = Qwen2ForCausalLM.from_pretrained(exported)
    assert restored.lm_head.weight.data_ptr() == restored.model.embed_tokens.weight.data_ptr()
    score(
        exported,
        path,
        records,
        tmp_path / "risk.json",
        cache=tmp_path / "teacher-cache",
        device="cpu",
        cap=4,
        token_chunk=2,
        vocab_chunk=3,
    )
    risk = json.loads((tmp_path / "risk.json").read_text())
    assert all(abs(row["loss"]) < 1e-7 for row in risk["rows"])
    hidden_analysis(cache, cache, cache, tmp_path / "hidden", rank=1)
    analyze_head(cache, cache, cache, tmp_path / "head.json", base_weights=path, teacher_weights=path, rank=1)
    assert all(row["identity_error"] == 0 for row in json.loads((tmp_path / "head.json").read_text())["rows"])


def test_omitted_tied_head_alias_has_identical_coordinates(tmp_path):
    embedding = torch.randn(7, 3)
    base = TensorCheckpoint(
        make_checkpoint(
            tmp_path / "base", {"model.embed_tokens.weight": embedding, "lm_head.weight": embedding.clone()}
        )
    )
    other = TensorCheckpoint(make_checkpoint(tmp_path / "other", {"model.embed_tokens.weight": embedding}))
    base.compatible(other)
    torch.testing.assert_close(other.get("lm_head.weight"), base.get("lm_head.weight"))


def test_weight_reference_cache_and_resume_validate_content(tmp_path, monkeypatch):
    from examples.mirroropd.research.studies.spectral.weights import analyze as weight_analysis

    key = "model.layers.0.mlp.up_proj.weight"
    generator = torch.Generator().manual_seed(61)
    b, dt, do = [torch.randn(5, 3, generator=generator, dtype=torch.float64) for _ in range(3)]
    paths = [
        make_checkpoint(tmp_path / name, {key: value})
        for name, value in (("base", b), ("teacher", b + dt), ("opd", b + do))
    ]
    calls = []
    original = torch.linalg.svdvals

    def counted(*args, **kwargs):
        calls.append(args[0].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svdvals", counted)
    cache = tmp_path / "references"
    weight_analysis(*paths, tmp_path / "first", cache=cache)
    cold_calls = len(calls)
    calls.clear()
    weight_analysis(*paths, tmp_path / "second", cache=cache)
    assert len(calls) < cold_calls
    first, second = [json.loads((tmp_path / name / "00000.json").read_text()) for name in ("first", "second")]
    for field in ("teacher", "opd", "controls", "missing", "base_frame_bands"):
        assert first[field] == second[field]
    assert not first["reference_cache_hit"] and second["reference_cache_hit"]
    calls.clear()
    weight_analysis(*paths, tmp_path / "second", cache=cache, resume=True)
    assert not calls
    # Same shape/path with changed bytes must never reuse the old teacher facts.
    save_file({key: b + 2 * dt}, paths[1] / "model.safetensors")
    weight_analysis(*paths, tmp_path / "changed", cache=cache)
    changed = json.loads((tmp_path / "changed/00000.json").read_text())
    assert not changed["reference_cache_hit"]
    with pytest.raises(ValueError):
        weight_analysis(*paths, tmp_path / "second", cache=cache, resume=True)


def test_precomputed_singular_values_preserve_exact_diagnostics():
    from examples.mirroropd.research.studies.spectral.decomposition import spectrum_summary

    x = torch.randn(9, 4, dtype=torch.float64)
    singular = torch.linalg.svdvals(x)
    assert spectrum_summary(x) == spectrum_summary(x, singular_values=singular)
    expected = random_controls(x, seed=19)
    actual = random_controls(x, seed=19, singular_values=singular)
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name])
