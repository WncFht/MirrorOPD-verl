import importlib.util
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "examples/mirroropd/benchmarks/vllm_sweep"
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location("mirroropd_sweep_runner", SCRIPTS / "run_sweep.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
finally:
    sys.path.pop(0)

PLAN = json.loads((SCRIPTS / "plan.json").read_text())


def test_queue_keeps_venv_python_path_instead_of_resolving_to_base_python(tmp_path):
    executable = tmp_path / "venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    args = SimpleNamespace(
        output=tmp_path / "out",
        plan=SCRIPTS / "plan.json",
        models=None,
        python=executable,
        model_root=tmp_path / "models",
        data_root=tmp_path / "data",
        action="dry-run",
    )
    queue = runner.Queue(args)
    assert queue.python == executable and queue.vllm == executable.with_name("vllm")


def argument(command, name):
    return command[command.index(name) + 1]


@pytest.mark.parametrize("profile", PLAN["models"], ids=lambda p: p["id"])
def test_full_sweep_preserves_all_requests_sampling_and_per_replica_resources(tmp_path, profile):
    command = runner.build_commands(
        PLAN,
        profile,
        profile["candidates"][1],
        tmp_path / "models with spaces",
        tmp_path / "all prompts.jsonl",
        tmp_path / "run",
        Path("/venv/vllm"),
        Path("/venv/python"),
    )
    serve = shlex.split(argument(command, "--serve-cmd"))
    bench = shlex.split(argument(command, "--bench-cmd"))
    assert argument(serve, "--data-parallel-size") == "4"
    assert argument(serve, "--tensor-parallel-size") == "1"
    assert int(argument(serve, "--max-model-len")) == profile["prompt_cap"] + 31744
    assert argument(bench, "--num-prompts") == "0" and argument(bench, "--request-rate") == "inf"
    assert "--ignore-eos" not in bench and argument(bench, "--custom-output-len") == "-1"
    assert "--skip-chat-template" in bench and "--disable-shuffle" in bench and "--no-oversample" in bench
    assert argument(bench, "--temperature") == "0.7" and argument(bench, "--top-p") == "0.9"
    extra = json.loads(argument(bench, "--extra-body"))
    assert extra["add_special_tokens"] is False
    assert extra.get("stop_token_ids", []) == profile["extra_stop_ids"]
    assert argument(command, "--num-runs") == "1" and "--show-stdout" in command
    assert "--logits-processors" not in serve and argument(serve, "--worker-cls").endswith("ProtocolWorker")
    assert (
        json.loads(argument(serve, "--compilation-config"))["cudagraph_capture_sizes"][-1]
        == profile["candidates"][1][0]
    )


def test_preflight_is_separate_from_full_measurements(tmp_path):
    command = runner.build_commands(
        PLAN,
        PLAN["models"][0],
        [32, 0.65, 8192],
        tmp_path,
        tmp_path / "data",
        tmp_path / "run",
        Path("/vllm"),
        Path("/python"),
        preflight=True,
    )
    bench = shlex.split(argument(command, "--bench-cmd"))
    assert argument(bench, "--num-prompts") == "4" and argument(bench, "--custom-output-len") == "64"


def test_native_custom_dataset_honors_explicit_jsonl_budget(tmp_path):
    from vllm.benchmarks.datasets import CustomDataset

    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"prompt": "2+2", "output_tokens": 31744}) + "\n")
    dataset = CustomDataset(dataset_path=str(path), random_seed=42, disable_shuffle=True)
    requests = dataset.sample(
        lambda text: SimpleNamespace(input_ids=[1, 2, 3]), num_requests=0, output_len=-1, skip_chat_template=True
    )
    assert len(requests) == 1 and requests[0].expected_output_len == 31744


def test_explicit_request_budget_overrides_base_model_default():
    from vllm.entrypoints.serve.utils.api_utils import get_max_tokens

    assert get_max_tokens(32768, 31744, 803, {"max_tokens": 2048}, None) == 31744


@pytest.mark.parametrize(
    "change", [{"completed": 9119}, {"failed": 1}, {"duration": 0}, {"output_throughput": float("nan")}]
)
def test_partial_or_failed_results_cannot_win(change):
    result = dict(completed=9120, failed=0, duration=123, output_throughput=9000, total_output_tokens=1107000)
    assert runner.eligible(result)
    assert not runner.eligible(result | change)


def test_ranking_uses_complete_runs_and_pooled_repeats():
    rows = [
        dict(status="complete", stage="a", candidate=[32, 0.65, 8192], duration=100, total_output_tokens=1000),
        dict(status="complete", stage="b", candidate=[64, 0.65, 8192], duration=80, total_output_tokens=1000),
        dict(status="complete", stage="repeat", candidate=[64, 0.65, 8192], duration=90, total_output_tokens=1000),
        dict(status="failed", stage="bad", candidate=[128, 0.8, 8192], duration=1, total_output_tokens=1000),
    ]
    best = runner.choose_best(rows)
    assert best["candidate"][0] == 64 and best["mean_duration"] == 85
    assert best["pooled_output_tps"] == pytest.approx(2000 / 170)


def test_export_repeats_inputs_without_leaking_ground_truth():
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return ",".join(map(str, ids))

        def encode(self, text, **kwargs):
            return list(map(int, text.split(",")))

    class Builder:
        def build_initial_tokens(self, messages):
            return [1, 2, 3]

    rows = [{"dataset": "MATH-500", "row": 0, "data": {"prompt": [], "reward_model": {"ground_truth": "ANSWER"}}}]
    requests, metadata = runner.prepare.__globals__["render_rows"](rows, Tokenizer(), Builder(), 1024, 16, 31744)
    assert len(requests) == 16 and all(r == {"prompt": "1,2,3", "output_tokens": 31744} for r in requests)
    assert metadata[0]["ground_truth"] == "ANSWER"
    with pytest.raises(ValueError, match="Overlong"):
        runner.prepare.__globals__["render_rows"](rows, Tokenizer(), Builder(), 2, 16, 31744)


def test_worker_preserves_verl_replica_seed_and_token_support(monkeypatch):
    import torch

    spec = importlib.util.spec_from_file_location("sweep_protocol_worker", SCRIPTS / "worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed = []
    monkeypatch.setattr(module.Worker, "__init__", lambda self, cfg, *a, **kw: observed.append(cfg.model_config.seed))
    cfg = SimpleNamespace(model_config=SimpleNamespace(seed=42), parallel_config=SimpleNamespace(data_parallel_index=3))
    module.ProtocolWorker(cfg)
    assert observed == [45]

    class Model:
        def compute_logits(self, hidden):
            return hidden.clone()

    model = Model()
    module.mask_tokenizer_tail(model, 3)
    result = model.compute_logits(torch.arange(8, dtype=torch.float32).reshape(2, 4))
    assert result[:, :3].tolist() == [[0, 1, 2], [4, 5, 6]]
    assert torch.isneginf(result[:, 3]).all()


def test_original_validation_rejects_materialized_failure_placeholders(tmp_path):
    records = tmp_path / "validation/records/0.jsonl"
    records.parent.mkdir(parents=True)
    records.write_text("{}\n" * 9120)
    data = {}
    for name, count in [("math500", 8000), ("amc23", 640), ("aime25", 480)]:
        data[f"eval/{name}/response_count"] = count
        data[f"eval/{name}/failed_count"] = 0
        data[f"eval/{name}/acc/mean"] = 0.5
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(json.dumps({"data": data}) + "\n")
    assert runner.validate_original_outputs(tmp_path)["valid"]
    data["eval/aime25/failed_count"] = 1
    metrics.write_text(json.dumps({"data": data}) + "\n")
    assert not runner.validate_original_outputs(tmp_path)["valid"]
