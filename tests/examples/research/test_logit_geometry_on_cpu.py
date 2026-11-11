# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import torch

from examples.mirroropd.research.studies.teacher_difference.logit_geometry import (
    LogitJacobian,
    leading_spectrum,
    request_projection,
    response_summary,
)
from examples.mirroropd.research.studies.teacher_difference.signals import probability_profile


def test_matrix_free_cross_response_matches_explicit_nonlinear_jacobians():
    torch.manual_seed(3)
    parameter = torch.randn(7, dtype=torch.float64, requires_grad=True)
    source_matrix, receiver_matrix = torch.randn(5, 7, dtype=torch.float64), torch.randn(5, 7, dtype=torch.float64)
    source_fn = lambda x: (source_matrix @ x).tanh()
    receiver_fn = lambda x: (receiver_matrix @ x).sin()
    source = LogitJacobian(source_fn(parameter), [parameter])
    receiver = LogitJacobian(receiver_fn(parameter), [parameter])
    js = torch.autograd.functional.jacobian(source_fn, parameter)
    jr = torch.autograd.functional.jacobian(receiver_fn, parameter)
    request, receiver_request = torch.randn(5, dtype=torch.float64), torch.randn(5, dtype=torch.float64)
    direction = source.pullback(request)
    actual = receiver.pushforward(direction)
    torch.testing.assert_close(actual, jr @ js.T @ request)
    torch.testing.assert_close(source.kernel(request), js @ js.T @ request)
    summary = response_summary(receiver_request, actual)
    assert abs(summary["request_response_inner"] - float(receiver_request @ jr @ js.T @ request)) < 1e-10
    assert parameter.grad is None


def test_ritz_spectrum_and_request_contributions_match_dense():
    torch.manual_seed(19)
    a = torch.randn(12, 12, dtype=torch.float64)
    kernel = a @ a.T
    result = leading_spectrum(lambda x: kernel @ x, a[0], iterations=12)
    expected = torch.linalg.eigvalsh(kernel).flip(0)
    torch.testing.assert_close(result["eigenvalues"], expected)
    assert max(result["residual_relative"]) < 1e-10
    request = torch.randn(12, dtype=torch.float64)
    projection = request_projection(request, result)
    assert 0 <= projection["unresolved_request_fraction"] <= 1


def test_probability_profile_preserves_pressure_and_native_support():
    z = torch.tensor([7.0, 0, -5, -15])
    q = torch.tensor([2.0, 1, 0, -1, 8])
    result = probability_profile(z, q)
    assert abs(sum(b["pressure"] for b in result["bins"]) - result["pressure"]) < 1e-8
    assert sum(b["tokens"] for b in result["bins"]) == len(z)
    assert result["native_support_floor"] > 0
    assert result["conditional_rkl"] < result["loss"]


def test_logit_operator_matches_tiny_qwen_parameter_gram():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from examples.mirroropd.research.studies.teacher_difference.logit_geometry import direction_inner
    from examples.mirroropd.research.studies.teacher_difference.runtime import prefix_logits

    torch.manual_seed(31)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=11,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            tie_word_embeddings=True,
            attention_dropout=0,
        )
    ).eval()
    model.set_attn_implementation("eager")
    row = dict(prompt_ids=[1, 2], response_ids=[3, 4])
    source = LogitJacobian(prefix_logits(model, row, 0), model.parameters())
    receiver = LogitJacobian(prefix_logits(model, row, 1), model.parameters())
    bs, br = torch.randn(11), torch.randn(11)
    gs, gr = source.pullback(bs), receiver.pullback(br)
    cross = receiver.pushforward(gs)
    torch.testing.assert_close(br @ cross, torch.tensor(direction_inner(gs, gr)), rtol=2e-5, atol=2e-5)
    self_response = source.kernel(bs)
    torch.testing.assert_close(bs @ self_response, torch.tensor(direction_inner(gs, gs)), rtol=2e-5, atol=2e-5)


def test_chunked_direction_dot_preserves_double_precision_and_inactive_parameters():
    from examples.mirroropd.research.studies.teacher_difference.logit_geometry import direction_inner
    from examples.mirroropd.research.studies.teacher_difference.metrics import tensor_inner

    a = torch.tensor([1e8, 1, -1e8, 3], dtype=torch.float64)
    b = torch.ones_like(a)
    expected = float(a @ b)
    assert abs(float(tensor_inner(a, b, chunk=2)) - expected) < 1e-12
    assert direction_inner((a, None), (b, None)) == expected
    assert abs(direction_inner((a,), (a,)) / float(a @ a) - 1) < 1e-14
    import pytest

    with pytest.raises(ValueError, match="length"):
        direction_inner((a,), (b, None))
    with pytest.raises(ValueError, match="shape"):
        tensor_inner(a, b[:2])


def test_checkpoint_store_excludes_duplicate_writer_and_rejects_changed_inputs(tmp_path):
    import pytest

    from examples.mirroropd.research.studies.teacher_difference.checkpoints import CheckpointStore

    root = tmp_path / "parts"
    with CheckpointStore(root, dict(prefixes=[("r1", 7)])) as store:
        store.save_tensors("site-000", dict(values=torch.arange(4)))
        store.save_json("pair-000", [dict(receiver=1, gain=-0.5)])
        with pytest.raises(BlockingIOError):
            with CheckpointStore(root, dict(prefixes=[("r1", 7)])):
                pass
    with CheckpointStore(root, dict(prefixes=[["r1", 7]])) as store:
        torch.testing.assert_close(store.tensors("site-000")["values"], torch.arange(4))
        assert store.json("pair-000")[0]["gain"] == -0.5
    with pytest.raises(ValueError, match="Stale"):
        with CheckpointStore(root, dict(prefixes=[("r1", 8)])):
            pass


def test_streamed_cross_panel_resume_matches_dense_and_reuses_symmetric_gram(tmp_path, monkeypatch):
    import pytest

    from examples.mirroropd.research.studies.teacher_difference import logit_probe
    from examples.mirroropd.research.studies.teacher_difference.checkpoints import CheckpointStore

    torch.manual_seed(41)
    model = torch.nn.Linear(3, 4, bias=False).double()
    rows = [
        dict(row_id=str(i), split="train" if i == 0 else "heldout", x=torch.randn(3, dtype=torch.float64))
        for i in range(3)
    ]

    def logits(student, row, position):
        return student(row["x"]).tanh()

    monkeypatch.setattr(logit_probe, "prefix_logits", logits)
    selected = [(row, 0) for row in rows]
    payloads = [dict(requests={"t/live": torch.randn(4, dtype=torch.float64)}) for _ in rows]
    dense_directions = [
        torch.autograd.grad((logits(model, row, 0) * p["requests"]["t/live"]).sum(), model.weight)[0].reshape(-1)
        for row, p in zip(rows, payloads, strict=True)
    ]
    actual_measure = logit_probe.measure_cross
    calls, gram_checks = [], []

    def counted(*args, **kwargs):
        calls.append(1)
        gram_checks.append(kwargs["gram"] is None)
        return actual_measure(*args, **kwargs)

    monkeypatch.setattr(logit_probe, "measure_cross", counted)
    with CheckpointStore(tmp_path, {"input": 1}) as store:
        save = store.save_json

        def interrupt_after_save(key, value):
            save(key, value)
            raise InterruptedError("Simulated preemption")

        monkeypatch.setattr(store, "save_json", interrupt_after_save)
        with pytest.raises(InterruptedError):
            logit_probe.measure_cross_panel(model, selected, [0, 1, 2], payloads, store, stage=0)
        monkeypatch.setattr(store, "save_json", save)
        result = logit_probe.measure_cross_panel(model, selected, [0, 1, 2], payloads, store, stage=0)
        assert len(calls) == 6  # Saved first pair is reused, not recalculated.
        assert sum(gram_checks) == 3  # One full-parameter dot per unordered pair.
        for item in result:
            expected = float(dense_directions[item["source"]] @ dense_directions[item["receiver"]])
            assert abs(item["gradient_inner"] - expected) < 1e-12
            assert abs(item["request_response_inner"] - expected) < 1e-12
        again = logit_probe.measure_cross_panel(model, selected, [0, 1, 2], payloads, store, stage=0)
        assert again == result
        assert len(calls) == 6


def test_shared_gram_accepts_memory_mapped_rows_and_progress(tmp_path):
    import numpy as np

    from examples.mirroropd.research.studies.teacher_difference.metrics import chunked_gram

    values = np.random.default_rng(9).normal(size=(4, 17)).astype(np.float32)
    paths = []
    for i, row in enumerate(values):
        path = tmp_path / f"{i}.npy"
        np.save(path, row)
        paths.append(path)
    rows = [np.load(path, mmap_mode="r") for path in paths]
    progress = []
    result = chunked_gram(rows, chunk=5, device="cpu", progress=lambda start, total: progress.append((start, total)))
    dense = torch.from_numpy(values).double()
    torch.testing.assert_close(result, dense @ dense.T)
    assert progress == [(0, 17), (5, 17), (10, 17), (15, 17)]


def test_lightweight_native_terms_preserve_metrics_and_loss_gradient():
    from examples.mirroropd.research.core.metrics import native_reverse_kl_terms, native_terms
    from examples.mirroropd.research.studies.teacher_difference.signals import position_signal, profile_and_signal

    torch.manual_seed(21)
    student = torch.randn(3, 13, requires_grad=True)
    teacher = torch.randn(3, 15)
    loss, lp, lq, p = native_reverse_kl_terms(student, teacher)
    broad, old_lp, old_lq, old_p, _ = native_terms(student, teacher)
    for actual, expected in zip((loss, lp, lq, p), (broad["rkl"], old_lp, old_lq, old_p), strict=True):
        torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(loss.sum(), student)[0]
    _, analytic = position_signal(student.detach(), teacher)
    torch.testing.assert_close(gradient, analytic)
    profile, derivative = profile_and_signal(student.detach()[0], teacher[0])
    torch.testing.assert_close(derivative, analytic[0])
    assert abs(profile["pressure"] - float(derivative.square().sum())) < 1e-7


def test_task_resume_keeps_completed_spectra_and_publishes_compatible_artifacts(tmp_path, monkeypatch):
    from examples.mirroropd.research.studies.teacher_difference import logit_probe

    torch.manual_seed(81)
    model = torch.nn.Linear(3, 5, bias=False)
    rows = [
        dict(row_id=str(i), split="train" if i == 0 else "heldout", domain="math", x=torch.randn(3)) for i in range(2)
    ]

    def logits(student, row, position):
        return student(row["x"]).tanh()

    monkeypatch.setattr(logit_probe, "prefix_logits", logits)
    monkeypatch.setattr(logit_probe, "load_model", lambda *a, **kw: model)
    monkeypatch.setattr(logit_probe, "collect_memory", lambda: None)
    selected = [(row, 0) for row in rows]
    requests = {"t": [dict(B0=torch.randn(5), q_logits=torch.randn(5)) for _ in rows]}
    task = dict(student="tiny", stage=0)
    signature = dict(protocol="cpu-test", prefixes=[("0", 0), ("1", 0)])
    args = dict(destination=tmp_path / "tiny.json", signature=signature, iterations=3, seed=42)
    actual_panel = logit_probe.measure_cross_panel

    def interrupted(*args, **kwargs):
        raise InterruptedError("Preempt after spectra")

    import pytest

    monkeypatch.setattr(logit_probe, "measure_cross_panel", interrupted)
    with pytest.raises(InterruptedError):
        logit_probe.run_task(task, selected, requests, [0, 1], **args)
    assert len(list((tmp_path / "tiny.parts").glob("site-*.pt"))) == 2

    def no_new_spectra(*args, **kwargs):
        raise AssertionError("Completed spectra must not be recomputed")

    monkeypatch.setattr(logit_probe, "site_payload", no_new_spectra)
    monkeypatch.setattr(logit_probe, "measure_cross_panel", actual_panel)
    logit_probe.run_task(task, selected, requests, [0, 1], **args)
    import json

    output = json.loads((tmp_path / "tiny.json").read_text())
    artifact = torch.load(tmp_path / "tiny.pt", weights_only=True)
    assert output["signature"] == json.loads(json.dumps(artifact["signature"]))
    assert len(output["sites"]) == 2 and len(output["cross"]) == 4
    assert len(artifact["spectra"]) == 2
    assert output["execution"]["direction_storage"] == "one-source-field-on-device"


def test_pressure_selection_is_shared_between_teachers_and_balances_questions():
    from examples.mirroropd.research.studies.teacher_difference.logit_probe import select_indices

    selected = []
    for split in ["train", "heldout"]:
        for question in range(4):
            row = dict(row_id=f"{split}-{question}", split=split, domain="math", response_ids=[1] * 9)
            selected.extend((row, pos) for pos in [0, 4, 8])
    requests = {"a": [], "b": []}
    for _, pos in selected:
        requests["a"].append(dict(B0=torch.tensor([10.0 if pos == 0 else 1.0])))
        requests["b"].append(dict(B0=torch.tensor([10.0 if pos == 8 else 1.0])))
    indices = select_indices(selected, requests, positions="pressure", max_sites=16)
    assert len(indices) == 16
    assert {selected[i][1] for i in indices} == {0, 8}
    assert len({selected[i][0]["row_id"] for i in indices[:8]}) == 8


def test_logit_report_shared_initial_and_explicit_pair_metadata(tmp_path):
    import json

    from examples.mirroropd.research.studies.teacher_difference.report_logit import collect, panel_summaries

    fields = {
        f"{teacher}/{mode}": dict(
            pressure=1.0,
            gradient_norm_sq=1.0,
            request_response_inner=1.0,
            leading_request_energy=[0.1, 0.2, 0.3],
            leading_accessibility_contribution=[0.3, 0.4, 0.3],
        )
        for teacher in ["justrl", "polaris"]
        for mode in ["live", "frozen_B0"]
    }
    sites = [
        dict(
            row_id=str(i),
            position=4,
            split="train" if i < 2 else "heldout",
            fields=fields,
            profiles={
                t: dict(loss=1.0, pressure=1.0, native_support_floor=0.0, bins=[dict(pressure=1.0)])
                for t in ["justrl", "polaris"]
            },
        )
        for i in range(4)
    ]
    edges = [
        dict(
            source=i,
            receiver=j,
            field=field,
            source_split=sites[i]["split"],
            receiver_split=sites[j]["split"],
            gradient_inner=0.25,
            request_response_inner=0.25,
            response_norm=1.0,
            request_response_cosine=0.25,
        )
        for i in range(4)
        for j in range(4)
        if i != j
        for field in fields
    ]
    spectrum = dict(
        eigenvalues=torch.tensor([3.0, 2.0, 1.0]),
        vectors=torch.eye(5)[:, :3],
        residual_relative=torch.zeros(3),
        projected_asymmetry=0.0,
    )
    signature = dict(prefixes=[("0", 4)])
    for branch, stage in [("justrl", 0), ("polaris", 240)]:
        path = tmp_path / f"{branch}-s{stage:03d}.json"
        path.write_text(
            json.dumps(dict(signature=signature, task=dict(target=branch, stage=stage), sites=sites, cross=edges))
        )
        torch.save(dict(signature=signature, spectra=[spectrum] * 4), path.with_suffix(".pt"))
    rows, cross, _ = collect(tmp_path)
    assert {r["branch"] for r in rows if r["stage"] == 0} == {"justrl", "polaris"}
    panels, transfers = panel_summaries(rows, cross)
    assert len(transfers) == 12
    for panel in panels:
        assert panel["C"] == 0.625
        assert panel["cross_record_normalized"] == 0.25


def test_question_logit_gram_is_invariant_to_duplicate_positions():
    import numpy as np

    from examples.mirroropd.research.studies.teacher_difference.analyze_logit import question_panel, transfer_moments

    def panel(replicas):
        vectors = torch.tensor([[1.0, 0.0], [1.0, 1.0], [-1.0, 1.0], [0.0, 2.0]], dtype=torch.float64)
        sites, gradients = [], []
        for question, count in enumerate(replicas):
            for _ in range(count):
                sites.append(
                    dict(
                        site=len(sites),
                        row_id=str(question),
                        pressure=1.0,
                        loss=2.0,
                        gradient_norm_sq=float(vectors[question].square().sum()),
                    )
                )
                gradients.append(vectors[question])
        edges = [
            dict(source=i, receiver=j, gradient_inner=float(a @ b))
            for i, a in enumerate(gradients)
            for j, b in enumerate(gradients)
            if i != j
        ]
        return question_panel(sites, edges)

    _, original, moments = panel([1, 1, 1, 1])
    _, repeated, duplicated = panel([1, 3, 2, 4])
    np.testing.assert_allclose(original, repeated)
    for key in moments:
        np.testing.assert_allclose(moments[key], duplicated[key])
    transfer = transfer_moments(original, np.full(4, 2.0), [0, 1], [2, 3])
    # Mean source=(1,.5); mean receiver=(-.5,1.5).
    assert abs(transfer["cross_inner"] - 0.25) < 1e-12
    assert abs(transfer["cross_over_source"] - 0.2) < 1e-12
    assert abs(transfer["cross_over_loss"] - 0.125) < 1e-12


def test_batched_prefix_vjp_matches_mean_of_individual_prefix_gradients():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from examples.mirroropd.research.studies.teacher_difference.logit_geometry import logit_pullback
    from examples.mirroropd.research.studies.teacher_difference.runtime import prefix_batch_logits, prefix_logits

    torch.manual_seed(81)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=13,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            tie_word_embeddings=True,
        )
    ).eval()
    prefixes = [
        (dict(prompt_ids=[1, 2], response_ids=[3, 4]), 0),
        (dict(prompt_ids=[2, 3, 1], response_ids=[4, 5, 6]), 2),
    ]
    request = torch.randn(2, 13)
    expected = [
        logit_pullback(prefix_logits(model, row, pos), model.parameters(), request[i])
        for i, (row, pos) in enumerate(prefixes)
    ]
    batched = prefix_batch_logits(model, prefixes)
    torch.testing.assert_close(
        batched, torch.stack([prefix_logits(model, row, pos) for row, pos in prefixes]), rtol=1e-5, atol=1e-6
    )
    actual = logit_pullback(batched, model.parameters(), request / 2)
    for i, value in enumerate(actual):
        torch.testing.assert_close(value, (expected[0][i] + expected[1][i]) / 2, rtol=1e-5, atol=1e-6)


def test_uniform_positions_select_independent_questions_deterministically():
    from examples.mirroropd.research.studies.teacher_difference.prefixes import select

    rows = [
        dict(row_id=str(i), domain="math" if i < 20 else "code", split="train", response_ids=list(range(100)))
        for i in range(30)
    ]
    kwargs = dict(per_stratum=100, seed=17, domains=["math"], position_mode="uniform")
    positions = select(rows, **kwargs)
    assert len(positions) == len({r["row_id"] for r, p in positions}) == 20
    assert positions == select(rows, **kwargs)
    assert len({p for r, p in positions}) > 10
    assert all(r["domain"] == "math" and 0 <= p < 100 for r, p in positions)


def test_clustered_transfer_bootstrap_keeps_signed_cross_signal():
    import numpy as np

    from examples.mirroropd.research.studies.teacher_difference.report_groups import transfer_bootstrap

    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    result = transfer_bootstrap(vectors @ vectors.T, [0, 1], [2, 3], np.ones(4), replicates=100)
    assert result["cross_inner"] == result["cross_inner_low"] == result["cross_inner_high"] == -1
    assert result["cross_cosine"] == -1
    assert result == transfer_bootstrap(vectors @ vectors.T, [0, 1], [2, 3], np.ones(4), replicates=100)


def test_streamed_group_gram_matches_dense_and_resumes(tmp_path, monkeypatch):
    from examples.mirroropd.research.studies.teacher_difference import group_probe
    from examples.mirroropd.research.studies.teacher_difference.checkpoints import CheckpointStore

    selected = [(dict(split="train" if i < 4 else "heldout", group=str(i // 2)), 0) for i in range(6)]
    vectors = torch.tensor(
        [[1.0, 0.0], [2.0, 1.0], [-1.0, 1.0], [0.0, 2.0], [3.0, -2.0], [1.0, 1.0]], dtype=torch.float64
    )
    calls = []

    def direction(student, selected, indices, payloads, field):
        calls.append(indices)
        return (vectors[indices].mean(0),)

    monkeypatch.setattr(group_probe, "grouped_direction", direction)
    payloads = [dict(requests={"teacher/live": torch.ones(2)}) for _ in selected]
    with CheckpointStore(tmp_path, dict(test=1)) as store:
        actual = group_probe.stream_group_gram(None, selected, payloads, store)
        calls.clear()
        resumed = group_probe.stream_group_gram(None, selected, payloads, store)
    means = vectors.reshape(3, 2, 2).mean(1)
    torch.testing.assert_close(actual["grams"]["teacher/live"], means @ means.T)
    torch.testing.assert_close(resumed["grams"]["teacher/live"], actual["grams"]["teacher/live"])
    assert calls == []


def test_large_panel_summary_uses_group_gram_and_prefix_pressure():
    import numpy as np

    from examples.mirroropd.research.studies.teacher_difference.report_groups import summarize

    vectors = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
    sites = [
        dict(fields={"teacher/live": dict(pressure=2.0, gradient_norm_sq=1.0)}, profiles={"teacher": dict(loss=4.0)})
        for _ in range(4)
    ]
    artifact = dict(
        groups=[dict(group=str(i), split="train" if i < 2 else "heldout", sites=[i]) for i in range(4)],
        grams={"teacher/live": vectors @ vectors.T},
        spectra={},
    )
    rows, transfers, spectra = summarize(dict(sites=sites, task=dict(stage=0)), artifact, "teacher", replicates=100)
    for row in rows:
        np.testing.assert_allclose(
            [row["P"], row["A"], row["C"], row["Xi"], row["PXi"], row["PXi_over_L"]], [2.0, 0.5, 1.0, 0.5, 1.0, 0.25]
        )
    assert transfers[0]["cross_inner"] == -1
    assert spectra == []


def test_future_transfer_excludes_previously_trained_batches(tmp_path):
    import json

    from examples.mirroropd.research.studies.teacher_difference.report_groups import summarize

    bank = [dict(group=str(i), batch_index=i, split="train" if i < 3 else "heldout") for i in range(6)]
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(bank))
    groups = [dict(group=str(i), split=bank[i]["split"], sites=[i]) for i in range(6)]
    sites = [
        dict(fields={"teacher/live": dict(pressure=1.0, gradient_norm_sq=1.0)}, profiles={"teacher": dict(loss=1.0)})
        for _ in range(6)
    ]
    _, transfers, _ = summarize(
        dict(sites=sites, task=dict(stage=16, records=str(path))),
        dict(groups=groups, grams={"teacher/live": torch.eye(6, dtype=torch.float64)}, spectra={}),
        "teacher",
        replicates=100,
    )
    assert [(r["source_role"], r["source_groups"]) for r in transfers] == [("all-train", 3), ("future-train", 2)]
