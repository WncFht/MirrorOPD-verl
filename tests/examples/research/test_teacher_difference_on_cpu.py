# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from examples.mirroropd.research.studies.teacher_difference.inputs import normalize_records, validate_records
from examples.mirroropd.research.studies.teacher_difference.metrics import batch_summary, gram_summary, pac_summary
from examples.mirroropd.research.studies.teacher_difference.pac import position_signal


def test_prefix_branch_selection_does_not_confuse_teacher_with_student_path():
    from examples.mirroropd.research.studies.teacher_difference.point_probe import branch_tasks

    native = dict(id="r1-polaris-b20", target="polaris", student="polaris-checkpoint")
    swapped = dict(id="r1-justrl-b20-target-polaris", target="polaris", student="justrl-checkpoint")
    explicit = dict(id="explicit-swap", branch="justrl", target="polaris", student="justrl-checkpoint")
    assert branch_tasks([native, swapped, explicit], "polaris") == [native]
    assert branch_tasks([native, swapped, explicit], "justrl") == [explicit]


def test_record_normalization_truncates_at_eos_and_audits_tokens():
    rows = normalize_records(
        [
            {"id": 7, "prompt_ids": [1, 2], "response_ids": [3, 4, 9, 8], "eos_token_id": 9},
            {"id": 8, "prompt_ids": [5, 6], "response_ids": [7, 8]},
        ],
        cap=10,
        bank_id="r1",
    )
    assert rows[0]["response_ids"] == [3, 4, 9]
    assert validate_records(rows) == {"records": 2, "response_tokens": 5}


def test_pac_and_batch_metrics_match_definitions():
    gradients = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    pac = pac_summary(pressure=[2.0, 2.0], gradient_vectors=gradients)
    assert pac["P"] == 2.0
    assert abs(pac["C"] - 0.5) < 1e-12
    assert abs(pac["rho"] - 1.0) < 1e-12
    batch = batch_summary(gradients)
    assert abs(batch["C_batch"] - 0.5) < 1e-12
    assert abs(batch["E_residual"] - 0.5) < 1e-12
    gram = gram_summary(gradients)
    assert gram["effective_rank"] == 2.0
    assert abs(gram["mean_offdiagonal_cosine"]) < 1e-12


def test_position_signal_is_zero_when_distributions_match():
    logits = torch.tensor([1.0, -0.5, 0.25], requires_grad=True)
    kl, b = position_signal(logits, logits.detach())
    assert abs(float(kl.detach())) < 1e-7
    assert torch.linalg.vector_norm(b) < 1e-7


def test_streaming_moments_and_chunked_gram_match_dense():
    from examples.mirroropd.research.studies.teacher_difference.metrics import chunked_gram

    torch.manual_seed(17)
    g = torch.randn(7, 19, dtype=torch.float64)
    norms = g.square().sum(1)
    dense = pac_summary(pressure=torch.arange(1, 8), gradient_vectors=g)
    stream = pac_summary(pressure=torch.arange(1, 8), gradient_vectors=g.mean(0)[None], gradient_norm_sq=norms)
    assert torch.allclose(chunked_gram(g, chunk=3), g @ g.T)
    for key in dense:
        assert abs(dense[key] - stream[key]) < 1e-10, key
    distinct = ((g @ g.T).sum() - norms.sum()) / 42
    assert abs(dense["cross_record_inner"] - distinct) < 1e-10
    orthogonal = pac_summary(pressure=[1, 1], gradient_vectors=torch.eye(2))
    assert orthogonal["cross_record_inner"] == 0
    assert orthogonal["cross_record_normalized"] == 0
    opposite = pac_summary(pressure=[1, 1], gradient_vectors=torch.tensor([[1.0], [-1.0]]))
    assert opposite["cross_record_inner"] == -1


@pytest.mark.parametrize("reduction", ["fixed_cap", "valid_tokens"])
def test_full_parameter_native_kernel_matches_dense_and_streaming(reduction):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from examples.mirroropd.research.studies.teacher_difference.pac import _flatten_grad, compute_batch
    from examples.mirroropd.research.studies.teacher_difference.runtime import response_hidden as _hidden

    torch.manual_seed(5)
    cfg = Qwen2Config(
        vocab_size=19,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        tie_word_embeddings=True,
    )
    student = Qwen2ForCausalLM(cfg).eval()
    teacher = Qwen2ForCausalLM(cfg).eval()
    rows = normalize_records(
        [
            {"id": 1, "prompt_ids": [1, 2], "response_ids": [3, 4]},
            {"id": 2, "prompt_ids": [5], "response_ids": [6, 7, 8]},
        ],
        cap=4,
    )
    hidden = {r["row_id"]: _hidden(teacher, r) for r in rows}
    exact = compute_batch(student, teacher.lm_head.weight.detach(), hidden, rows, cap=4, reduction=reduction)
    for i, row in enumerate(rows):
        student.zero_grad(set_to_none=True)
        ids = torch.tensor([row["prompt_ids"] + row["response_ids"][:-1]])
        z = student(ids).logits[0, len(row["prompt_ids"]) - 1 :]
        with torch.no_grad():
            zq = teacher(ids).logits[0, len(row["prompt_ids"]) - 1 :]
        lp, lq = z.log_softmax(-1), zq.log_softmax(-1)
        loss = (lp.exp() * (lp - lq)).sum() / (4 if reduction == "fixed_cap" else len(row["response_ids"]))
        loss.backward()
        torch.testing.assert_close(exact["gradient_vectors"][i], _flatten_grad(student), rtol=2e-4, atol=2e-7)
        assert abs(exact["losses"][i] - loss.item()) < 1e-6
    stream = compute_batch(
        student, teacher.lm_head.weight.detach(), hidden, rows, cap=4, retain_samples=False, reduction=reduction
    )
    torch.testing.assert_close(stream["gradient_vectors"][0], exact["gradient_vectors"].mean(0))

    torch.testing.assert_close(stream["gradient_norm_sq"], exact["gradient_norm_sq"], rtol=1e-6, atol=1e-10)
    student.train()
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    checkpointed = compute_batch(
        student, teacher.lm_head.weight.detach(), hidden, rows, cap=4, retain_samples=False, reduction=reduction
    )
    torch.testing.assert_close(checkpointed["gradient_vectors"], stream["gradient_vectors"])
    torch.testing.assert_close(checkpointed["gradient_norm_sq"], stream["gradient_norm_sq"])


def test_position_derivative_keeps_teacher_extra_support():
    torch.manual_seed(7)
    z = torch.randn(3, 5, requires_grad=True)
    q = torch.randn(3, 7)
    kl, b = position_signal(z, q)
    (gradient,) = torch.autograd.grad(kl.sum(), z)
    torch.testing.assert_close(b, gradient)
    conditional = (z.softmax(-1) * (z.log_softmax(-1) - q[:, :5].log_softmax(-1))).sum(-1)
    assert torch.all(kl > conditional)


def test_span_upper_bound_distinguishes_cancellation_from_missing_span():
    import numpy as np

    from examples.mirroropd.research.studies.teacher_difference.geometry import span_diagnostics

    vectors = np.array([[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    gram = vectors @ vectors.T
    out = span_diagnostics(gram, [0, 1], [0.0, 0.0, 1.0])
    assert abs(out["span_cosine_upper"] - 2**-0.5) < 1e-10
    assert abs(out["outside_span_energy"] - 0.5) < 1e-10
    assert out["span_cosine_upper"] + 1e-12 >= out["uniform_cosine"]
    assert abs(out["oracle_positive_cosine"] - out["span_cosine_upper"]) < 1e-10


def test_fisher_attenuation_and_native_support_floor_identity():
    from examples.mirroropd.research.studies.teacher_difference.pac import fisher_diagnostics

    z = torch.tensor([[1.0, -1.0, 0.5]])
    q = torch.tensor([[-1.0, 0.3, 2.0, 4.0]])
    out = fisher_diagnostics(z, q)
    lp, lq = z.log_softmax(-1), q.log_softmax(-1)
    p = lp.exp()[0]
    fisher = torch.diag(p) - p[:, None] * p[None, :]
    residual = lp[0] - lq[0, :3]
    torch.testing.assert_close(out["pressure"][0], (fisher @ residual).square().sum())
    assert 0 <= out["fisher_request_gain"][0] <= 0.5
    kl, _ = position_signal(z, q)
    torch.testing.assert_close(kl, out["conditional_rkl"] + out["native_support_floor"])
    conditional_z = q[:, :3].clone().requires_grad_(True)
    minimum, derivative = position_signal(conditional_z, q)
    assert derivative.norm() < 1e-6
    torch.testing.assert_close(minimum, fisher_diagnostics(conditional_z, q)["native_support_floor"])


def test_cross_receiver_correction_removes_orthogonal_receiver_noise():
    import numpy as np

    from examples.mirroropd.research.studies.teacher_difference.geometry import cross_receiver_projection

    # Four receivers share e0 and have mutually orthogonal idiosyncratic parts.
    vectors = np.zeros((6, 5))
    vectors[:2, 0] = 1
    vectors[2:, 0] = 1
    vectors[2:, 1:] = np.eye(4) * 3
    gram = vectors @ vectors.T
    result = cross_receiver_projection(gram, [0, 1], [2, 3, 4, 5])
    assert abs(result["receiver_shared_inner"] - 1) < 1e-10
    assert abs(result["projected_shared_fraction"] - 1) < 1e-10
    assert result["empirical_projected_mean_norm_sq"] / result["empirical_receiver_mean_norm_sq"] < 0.4
    pair = cross_receiver_projection(gram, [0, 1], [2, 3])
    assert abs(pair["projected_shared_fraction"] - 1) < 1e-10
    assert np.isnan(pair["fraction_jackknife_se"])


def test_compiled_matrix_tracks_inputs_and_rejects_changed_bank(tmp_path):
    import json

    import pytest

    from examples.mirroropd.research.studies.teacher_difference.batch import compile_plan, status
    from examples.mirroropd.research.studies.teacher_difference.stages import run

    for name in ["student", "teacher"]:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        (directory / "model.safetensors").write_bytes(b"identity-only fixture")
    bank = tmp_path / "bank.json"
    row = dict(id=1, prompt_ids=[1, 2], response_ids=[3], split="heldout", domain="math")
    bank.write_text(json.dumps([row]))
    plan = dict(
        models=dict(student="student", target="teacher"),
        banks=dict(initial=dict(path="bank.json", cap=4)),
        series=[
            dict(
                id="test",
                model="tiny",
                targets=["target"],
                banks=["initial"],
                checkpoints=[dict(step=0, student="student"), dict(step=16, student="student")],
            )
        ],
    )
    manifest = compile_plan(plan, base=tmp_path, gpus=[0, 5])
    assert len(manifest["tasks"]) == 2
    assert all(task["lane"] == 0 for task in manifest["tasks"])
    assert status(manifest, tmp_path)["completed"] == 0
    plan["defaults"] = dict(reduction="valid_tokens", source_policy="independent")
    revised = compile_plan(plan, base=tmp_path, gpus=[0, 5])
    assert revised["tasks"][0]["reduction"] == "valid_tokens"
    assert revised["tasks"][0]["source_policy"] == "independent"
    assert revised["tasks"][0] != manifest["tasks"][0]
    plan["defaults"]["reduction"] = "typo"
    with pytest.raises(ValueError, match="reduction"):
        compile_plan(plan, base=tmp_path, gpus=[0, 5])
    plan.pop("defaults")
    row["response_ids"] = [4]
    bank.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="Inputs changed after plan compilation"):
        run(manifest["tasks"][0], tmp_path)
    plan["series"] *= 2
    with pytest.raises(ValueError, match="duplicate task"):
        compile_plan(plan, base=tmp_path, gpus=[0, 5])


def test_checkpoint_preflight_rejects_deleted_indexed_shards(tmp_path):
    import json

    import pytest

    from examples.mirroropd.research.studies.teacher_difference.runtime import checkpoint_identity

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map={"w": "missing.safetensors"})))
    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        checkpoint_identity(tmp_path)


def test_shared_loader_and_teacher_cache_preserve_values_and_invalidate(tmp_path):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from examples.mirroropd.research.studies.teacher_difference.runtime import (
        checkpoint_identity,
        load_model,
        response_hidden,
        teacher_cache,
    )

    cfg = Qwen2Config(
        vocab_size=19,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        tie_word_embeddings=True,
    )
    original = Qwen2ForCausalLM(cfg).eval()
    directory = tmp_path / "model"
    original.save_pretrained(directory)
    model = load_model(directory, device="cpu")
    row = normalize_records([dict(id=1, prompt_ids=[1, 2], response_ids=[3, 4])])[0]
    torch.testing.assert_close(response_hidden(model, row), response_hidden(original, row))
    cache = teacher_cache(tmp_path / "cache", directory, [row], device="cpu")
    torch.testing.assert_close(cache["hidden"]["1"], response_hidden(model, row))
    repeated = teacher_cache(tmp_path / "cache", directory, [row], device="cpu")
    assert cache["identity"] == repeated["identity"]
    before = checkpoint_identity(directory)
    with torch.no_grad():
        original.model.layers[0].mlp.down_proj.weight.add_(0.5)
    original.save_pretrained(directory)
    assert checkpoint_identity(directory) != before
    changed = teacher_cache(tmp_path / "cache", directory, [row], device="cpu")
    assert changed["identity"] != cache["identity"]
    assert len(list((tmp_path / "cache/teacher-v1").glob("*.pt"))) == 2


def test_sweep_resumes_only_unfinished_lanes_and_checks_completion(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    from examples.mirroropd.research.studies.teacher_difference.runtime import write_json

    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "config.json").write_text("{}")
    (directory / "model.safetensors").write_bytes(b"preflight fixture")
    manifest = tmp_path / "manifest.json"
    tasks = [dict(id=f"task-{lane}", lane=lane, student=str(directory), teacher=str(directory)) for lane in [0, 5]]
    write_json(manifest, dict(tasks=tasks))
    write_json(tmp_path / "results/task-0/result.json", dict(finished=True, signature=dict(task=tasks[0])))
    calls = tmp_path / "calls.jsonl"
    broker = tmp_path / "broker.py"
    broker.write_text(f"import sys,json\nwith open({str(calls)!r},'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n")
    guard = tmp_path / "guard.py"
    guard.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "manifest=Path(sys.argv[sys.argv.index('--manifest')+1])\n"
        "lane=int(sys.argv[sys.argv.index('--lane')+1])\n"
        "for task in json.loads(manifest.read_text())['tasks']:\n"
        " if task['lane']!=lane: continue\n"
        " p=manifest.parent/'results'/task['id']/'result.json'\n"
        " p.parent.mkdir(parents=True,exist_ok=True)\n"
        " p.write_text(json.dumps(dict(finished=True,signature=dict(task=task))))\n"
    )
    command = [
        sys.executable,
        "-m",
        "examples.mirroropd.research.studies.teacher_difference",
        "sweep",
        "--manifest",
        str(manifest),
        "--repo",
        str(Path.cwd()),
        "--broker",
        str(broker),
        "--guard",
        str(guard),
        "--gpus",
        "0",
        "5",
    ]
    subprocess.run(command, check=True, capture_output=True)
    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
    assert [row[0] for row in recorded] == ["reserve", "attach"]
    assert all(row[row.index("--gpu") + 1] == "5" for row in recorded)
    result = json.loads((tmp_path / "sweep-exit.json").read_text())
    assert result["completed"] == result["planned"] == 2
    subprocess.run(command, check=True, capture_output=True)
    assert len(calls.read_text().splitlines()) == 2
    (tmp_path / "results/task-5/result.json").unlink()
    guard.write_text("pass\n")
    failed = subprocess.run(command, capture_output=True)
    assert failed.returncode == 1
    assert json.loads((tmp_path / "sweep-exit.json").read_text())["completed"] == 1


def test_joint_analysis_preserves_negative_transfer_and_paired_prefix_ratios():
    import numpy as np

    from examples.mirroropd.research.studies.teacher_difference.analyze_transfer import gram_stats, prefix_ratios

    gradients = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0]])
    data = {"geometry": {"gram": (gradients @ gradients.T).tolist()}}
    stats = gram_stats(data, [0, 1])
    assert stats["self_energy"] == 1
    assert stats["cross_energy"] == -1
    assert stats["cross_coherence"] == -1
    assert gram_stats(data, [2])["cross_energy"] is None

    def probe(live_pressure, live_norm, frozen_norm):
        return {
            "fields": {
                "teacher/live": {"pressure": live_pressure, "gradient_norm_sq": live_norm},
                "teacher/frozen_B0": {"pressure": [1.0, 3.0], "gradient_norm_sq": frozen_norm},
            }
        }

    initial = probe([1.0, 3.0], [2.0, 6.0], [2.0, 6.0])
    final = probe([0.5, 1.5], [2.0, 6.0], [4.0, 12.0])
    # Resampled repeated sites must retain their multiplicity in both endpoints.
    assert np.allclose(prefix_ratios(initial, final, "teacher", [0, 0, 1]), [0.5, 2.0, 2.0])


@pytest.mark.parametrize("reduction,denominator", [("fixed_cap", 4), ("valid_tokens", 2)])
def test_trajectory_frozen_vjp_matches_dense_tied_head_and_scaling(reduction, denominator):
    from examples.mirroropd.research.studies.teacher_difference.requests import trajectory_projection
    from verl.trainer.distillation.numerics import kl_rows

    torch.manual_seed(31)
    # The current head also creates hidden states: exercise tied-parameter accumulation.
    w = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    x = torch.randn(2, 7, dtype=torch.float64)
    h = x @ w
    h0, w0 = torch.randn_like(h), torch.randn_like(w)
    ht, wt = torch.randn_like(h), torch.randn(9, 3, dtype=torch.float64)
    initial_loss, b0 = kl_rows(h0 @ w0.T, ht @ wt.T)
    live_loss, live = kl_rows(h @ w.T, ht @ wt.T)
    expected = torch.autograd.grad(((h @ w.T) * b0.detach()).sum() / denominator, w, retain_graph=True)[0]
    result = trajectory_projection(h, w, ht, wt, initial=(h0, w0), cap=4, token_chunk=1, reduction=reduction)
    actual = torch.autograd.grad(result[0], w, retain_graph=True)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(result[1], live_loss)
    torch.testing.assert_close(result[3][:, 0], live.square().sum(-1))
    torch.testing.assert_close(result[3][:, 1], b0.square().sum(-1))
    torch.testing.assert_close(result[3][:, 2], (live * b0).sum(-1))
    torch.testing.assert_close(result[3][:, 3], initial_loss)
    native = trajectory_projection(h, w, ht, wt, cap=4, token_chunk=1, reduction=reduction)
    native_g = torch.autograd.grad(native[0], w, retain_graph=True)[0]
    dense_g = torch.autograd.grad(live_loss.sum() / denominator, w)[0]
    torch.testing.assert_close(native_g, dense_g)
