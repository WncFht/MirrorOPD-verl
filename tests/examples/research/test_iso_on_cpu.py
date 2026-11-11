# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from examples.mirroropd.research.studies.spectral.iso.math import compose, polar, tangent, transport


@pytest.mark.parametrize("shape", [(7, 7), (5, 9), (9, 5)])
def test_transport_feasibility_and_identity(shape):
    torch.manual_seed(41)
    base = torch.randn(*shape, dtype=torch.float64)
    target = base + 0.02 * torch.randn_like(base)
    f0 = torch.linalg.svd(base, full_matrices=False)
    f1 = torch.linalg.svd(target, full_matrices=False)
    row = transport(f0, f1, 0.9)
    assert row["admissible"]
    assert all(0 <= v <= 1 + 1e-10 for v in row["unexplained"].values())
    assert row["unexplained"]["mix"] >= row["unexplained"]["left"] - 1e-10
    assert row["unexplained"]["mix"] >= row["unexplained"]["right"] - 1e-10
    same = transport(f0, f0, 0.9)
    assert max(same["errors"].values()) < 1e-12


def test_merger_preserves_base_spectrum_and_sign_invariance():
    torch.manual_seed(7)
    base = torch.randn(9, 5, dtype=torch.float64)
    f0 = torch.linalg.svd(base, full_matrices=False)
    f1 = torch.linalg.svd(base + torch.randn_like(base) * 0.03, full_matrices=False)
    value, _ = compose(f0, [f1])
    torch.testing.assert_close(torch.linalg.svdvals(value), f0[1])
    signs = torch.tensor([1, -1, 1, -1, 1], dtype=torch.float64)
    changed = (f1[0] * signs, f1[1], f1[2] * signs[:, None])
    value2, _ = compose(f0, [changed])
    torch.testing.assert_close(value, value2)
    x = polar(torch.randn(9, 5, dtype=torch.float64))
    t = tangent(x, torch.randn_like(x))
    torch.testing.assert_close(x.T @ t + t.T @ x, torch.zeros(5, 5, dtype=x.dtype), atol=1e-12, rtol=0)


@pytest.mark.parametrize("offload", [False, True])
def test_factor_adam_preserves_fixed_spectrum_and_spectrum_only_frames(offload):
    from examples.mirroropd.research.studies.spectral.iso.training import FactorAdam

    torch.manual_seed(23)
    value = torch.randn(7, 5)
    for mode in ("frames", "spectrum"):
        parameter = torch.nn.Parameter(value.clone())
        # CPU reference uses the same factor updates; GPU SVD helper is replaced only for initialization.
        import unittest.mock

        from examples.mirroropd.research.studies.spectral.scan import thin_svd

        with unittest.mock.patch(
            "examples.mirroropd.research.studies.spectral.iso.training.thin_svd",
            side_effect=lambda x, **kw: thin_svd(x),
        ):
            optimizer = FactorAdam([("matrix", parameter)], 1e-3, mode)
        optimizer.offload = offload
        initial = torch.linalg.svdvals(parameter.detach())
        u0, _, vh0 = torch.linalg.svd(parameter.detach(), full_matrices=False)
        for _ in range(3):
            parameter.grad = torch.randn_like(parameter)
            optimizer.step()
        if mode == "frames":
            torch.testing.assert_close(torch.linalg.svdvals(parameter.detach()), initial, rtol=2e-6, atol=2e-6)
        else:
            coordinates = u0.T @ parameter.detach() @ vh0.T
            torch.testing.assert_close(coordinates, torch.diag(coordinates.diagonal()), rtol=0, atol=3e-6)


def test_extension_queue_creates_separate_module_result(tmp_path):
    import hashlib
    import json

    from examples.mirroropd.research.studies.spectral.absorption.queue import worker

    source = tmp_path / "source"
    source.mkdir()
    script = source / "iso_queue_smoke.py"
    script.write_text(
        "import pathlib,sys\n"
        "output=pathlib.Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=False)\n"
        "(output/'complete.json').write_text('{}')\n"
    )
    (tmp_path / "source-hashes.json").write_text(
        json.dumps({script.name: hashlib.sha256(script.read_bytes()).hexdigest()})
    )
    (tmp_path / "settings.json").write_text(json.dumps(dict(stop_paths=[])))
    output = tmp_path / "attempt-001"
    task = dict(
        task_id=0,
        job_id=0,
        phase="extension",
        module="iso_queue_smoke",
        arguments=["{attempt}/result"],
        output_attempt=str(output),
        complete_marker="result/complete.json",
        status="pending",
    )
    (tmp_path / "queue.json").write_text(json.dumps(dict(jobs=[task])))
    worker(tmp_path, 0)
    assert json.loads((tmp_path / "queue.json").read_text())["jobs"][0]["status"] == "complete"
    assert (output / "result/complete.json").is_file()
    assert (output / "command.log").is_file()
    assert json.loads((output / "exit.json").read_text())["complete"]


def test_analysis_uses_active_queue_instead_of_superseded_queue(tmp_path):
    import json

    from examples.mirroropd.research.studies.spectral.iso.analyze import summarize

    (tmp_path / "offline-manifest.json").write_text(json.dumps(dict(jobs=[])))
    active = tmp_path / "offline-v2"
    active.mkdir()
    (active / "queue.json").write_text(json.dumps(dict(jobs=[dict(status="pending", label="future")])))
    (tmp_path / "active-queue.json").write_text(json.dumps(dict(path=str(active))))
    summary = summarize(tmp_path, tmp_path / "analysis")
    assert summary["queue_root"] == str(active)
    assert summary["queue_counts"] == dict(pending=1)
    assert not summary["all_tasks_complete"]
    successor = tmp_path / "scale-controls-v1"
    successor.mkdir()
    (successor / "queue.json").write_text(json.dumps(dict(jobs=[dict(status="pending", label="scale")])))
    (tmp_path / "queue-chain.json").write_text(json.dumps(dict(paths=[str(active), str(successor)])))
    chained = summarize(tmp_path, tmp_path / "analysis")
    assert chained["queue_counts"] == dict(pending=2)
    assert chained["queue_roots"] == [str(active), str(successor)]


@pytest.mark.parametrize("scale_only", [True, False])
def test_saved_spectra_recover_radial_projection(scale_only):
    from examples.mirroropd.research.studies.spectral.iso.analyze import radial_statistics

    torch.manual_seed(71)
    base = torch.randn(9, 5, dtype=torch.float64)
    target = base * 0.99 if scale_only else base + 0.03 * torch.randn_like(base)
    s0, s1 = torch.linalg.svdvals(base), torch.linalg.svdvals(target)
    delta = target - base
    facts = dict(
        base_energy=float(base.square().sum()),
        update_energy=float(delta.square().sum()),
        singular_values=s1.tolist(),
        spectral_distance=float((s1 - s0).norm()),
    )
    recovered = radial_statistics(facts, s0.tolist())
    direct = float((base * delta).sum().square() / base.square().sum() / delta.square().sum())
    assert recovered["radial_update_fraction"] == pytest.approx(direct, abs=1e-10)
    if scale_only:
        assert recovered["radial_update_fraction"] == pytest.approx(1)
        assert recovered["best_spectrum_scale"] == pytest.approx(0.99)
        assert recovered["spectrum_after_scale_fraction"] < 1e-10


def test_scale_and_shape_interventions_have_distinct_invariants():
    from examples.mirroropd.research.studies.spectral.iso.math import spectrum_edit

    torch.manual_seed(64)
    base = torch.randn(7, 5, dtype=torch.float64)
    target = base * 1.2 + torch.randn_like(base) * 0.1
    s0 = torch.linalg.svdvals(base)
    u1, s1, vh1 = torch.linalg.svd(target, full_matrices=False)
    restore = (u1 * s0) @ vh1 - target
    ratio = float(s1.norm() / s0.norm())
    norm = spectrum_edit(target, restore, None, ratio, "norm-only", 1)
    shape = spectrum_edit(target, restore, None, ratio, "shape-only", 1)
    torch.testing.assert_close(norm.norm(), base.norm())
    torch.testing.assert_close(shape.norm(), target.norm())
    torch.testing.assert_close(torch.linalg.svdvals(norm) / norm.norm(), s1 / s1.norm())
    torch.testing.assert_close(torch.linalg.svdvals(shape) / shape.norm(), s0 / s0.norm())
    torch.testing.assert_close(spectrum_edit(target, restore, None, ratio, "restore", 1), target)


def test_completed_functional_analysis_ignores_progress_sidecars(tmp_path):
    import json

    from examples.mirroropd.research.studies.spectral.iso.analyze import summarize

    (tmp_path / "offline-manifest.json").write_text(json.dumps(dict(jobs=[])))
    queue = tmp_path / "offline-v2"
    queue.mkdir()
    attempt = queue / "attempt-001"
    scores = attempt / "result/scores"
    scores.mkdir(parents=True)
    condition = dict(role="student", kind="restore", alpha=1.0)
    row = dict(origin="initial-student", row_id="math:1", split="heldout", domain="math", rkl=0.1, fkl=0.2)
    (scores / "student-restore-alpha-1.0.json").write_text(json.dumps(dict(condition=condition, rows=[row])))
    (scores / "student-restore-alpha-1.0.progress.json").write_text(json.dumps(dict(condition=condition, completed=1)))
    task = dict(
        status="complete", label="functional-7", output_attempt=str(attempt), complete_marker="result/complete.json"
    )
    (queue / "queue.json").write_text(json.dumps(dict(jobs=[task])))
    (tmp_path / "active-queue.json").write_text(json.dumps(dict(path=str(queue))))
    summary = summarize(tmp_path, tmp_path / "analysis")
    assert len(summary["functional"]) == 1
    assert summary["functional"][0]["rkl"]["delta"] == 0


def test_native_online_kl_matches_dense_loss_and_hidden_gradient():
    from examples.mirroropd.research.studies.spectral.iso.training import online_loss

    torch.manual_seed(91)
    hs = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    ht = torch.randn_like(hs)
    ws = torch.randn(13, 5, dtype=torch.float64)
    wt = torch.randn_like(ws)
    ls, lt = (hs @ ws.T).log_softmax(-1), (ht @ wt.T).log_softmax(-1)
    expected = (ls.exp() * (ls - lt)).sum() / 128
    expected_gradient = torch.autograd.grad(expected, hs)[0]
    actual = online_loss(hs, ws, ht, wt, token_chunk=3)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, hs)[0], expected_gradient)


def test_training_contract_rejects_diagnostic_fallback_and_heldout_overlap(tmp_path):
    import json

    from examples.mirroropd.research.studies.spectral.iso.training import load_training_prompts

    path = tmp_path / "prompts.json"
    row = dict(row_id="dapo:1", prompt_ids=[1, 2], domain="math", eos_token_id=0)
    path.write_text(json.dumps(dict(source_family="diagnostic", source="bank", rows=[row])))
    with pytest.raises(ValueError, match="source_family=dapo"):
        load_training_prompts(path, {})
    path.write_text(json.dumps(dict(source_family="dapo", source="train.parquet", rows=[row])))
    with pytest.raises(ValueError, match="overlap"):
        load_training_prompts(path, {"initial-student": [dict(row, split="heldout")]})
    rows, source = load_training_prompts(path, {})
    assert rows == [row] and source["count"] == 1


def test_continuation_only_accepts_audited_setting_cancellations(tmp_path):
    import json

    from examples.mirroropd.research.studies.spectral.iso.continue_queue import predecessor_ready

    predecessor = tmp_path / "queue"
    task = dict(
        task_id=3,
        label="training-7-frames-42",
        status="failed",
        output_attempt=str(predecessor / "tasks/training-7-frames-42/attempt-001"),
    )
    assert not predecessor_ready(tmp_path, predecessor, [task])
    (tmp_path / "training-setting-correction.json").write_text(json.dumps(dict(cancelled_children=[dict(task=task)])))
    assert predecessor_ready(tmp_path, predecessor, [task])
    assert not predecessor_ready(tmp_path, predecessor, [dict(task, output_attempt="another-attempt")])
