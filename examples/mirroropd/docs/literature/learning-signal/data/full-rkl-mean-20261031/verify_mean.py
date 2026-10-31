"""CPU checks for KL reduction; execute repository agg_loss without its optional imports."""

import ast
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def main():
    torch.set_num_threads(1)
    root = next(
        parent for parent in Path(__file__).resolve().parents if (parent / "verl/trainer/ppo/core_algos.py").is_file()
    )
    namespace = {"torch": torch}
    masked_sum = load_function(root / "verl/utils/torch_functional.py", "masked_sum", namespace)
    namespace["verl_F"] = SimpleNamespace(masked_sum=masked_sum)
    aggregate = load_function(root / "verl/trainer/ppo/core_algos.py", "agg_loss", namespace)
    results = {"device": "cpu", "torch": torch.__version__, "aggregation_source": "verl/trainer/ppo/core_algos.py"}

    losses = torch.tensor([[1.0] * 2 + [0.0] * 6, [3.0] * 8], dtype=torch.float64)
    mask = torch.tensor([[1.0] * 2 + [0.0] * 6, [1.0] * 8], dtype=torch.float64)
    token = aggregate(losses, mask, "token-mean").item()
    sequence = aggregate(losses, mask, "seq-mean-token-mean").item()
    assert abs(token - 2.6) < 1e-12
    # Upstream's per-sequence division uses a 1e-8 numerical epsilon.
    assert abs(sequence - 2.0) < 1e-8
    results["length_example"] = {"token_mean": token, "sequence_mean": sequence, "naive_rank_mean": 2.0}

    torch.manual_seed(41)
    shape = (4, 5, 7)
    student = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn(shape, dtype=torch.float64)
    log_p, log_q = student.log_softmax(-1), teacher.log_softmax(-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1)
    reference = kl.sum()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mean = F.kl_div(log_q, log_p, log_target=True, reduction="mean")
    batchmean = F.kl_div(log_q, log_p, log_target=True, reduction="batchmean")
    flat_batchmean = F.kl_div(log_q.reshape(-1, 7), log_p.reshape(-1, 7), log_target=True, reduction="batchmean")
    torch.testing.assert_close(mean, reference / (4 * 5 * 7))
    torch.testing.assert_close(batchmean, reference / 4)
    torch.testing.assert_close(flat_batchmean, reference / (4 * 5))
    torch.testing.assert_close(
        torch.autograd.grad(batchmean, student, retain_graph=True)[0],
        torch.autograd.grad(reference / 4, student, retain_graph=True)[0],
    )
    results["pytorch_reductions"] = {
        "shape": list(shape),
        "sum": reference.item(),
        "mean": mean.item(),
        "batchmean": batchmean.item(),
        "flattened_batchmean": flat_batchmean.item(),
        "student_target_gradient_preserved": True,
    }

    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.float64)
    errors = {}
    for mode in ("token-mean", "seq-mean-token-mean", "seq-mean-token-sum"):
        kwargs = {"batch_num_tokens": mask.sum().item(), "global_batch_size": 4}
        expected = aggregate(kl, mask, mode, **kwargs)
        # Two simulated DP ranks, each with two one-sequence micro-batches;
        # averaging rank gradients models FSDP's DP reduction.
        pieces = [aggregate(kl[i : i + 1], mask[i : i + 1], mode, dp_size=2, **kwargs) for i in range(4)]
        reduced = sum(pieces) / 2
        expected_gradient = torch.autograd.grad(expected, student, retain_graph=True)[0]
        actual_gradient = torch.autograd.grad(reduced, student, retain_graph=True)[0]
        torch.testing.assert_close(reduced, expected)
        torch.testing.assert_close(actual_gradient, expected_gradient)
        errors[mode] = {
            "loss": expected.item(),
            "gradient_max_error": (actual_gradient - expected_gradient).abs().max().item(),
        }
    results["simulated_dp_and_microbatch"] = errors

    lengths = mask.sum(-1)
    weighted = (kl * mask / lengths[:, None]).sum() / 4
    sequence_mean = aggregate(kl, mask, "seq-mean-token-mean", global_batch_size=4)
    torch.testing.assert_close(weighted, sequence_mean, atol=1e-9, rtol=1e-8)
    gradient = torch.autograd.grad(weighted, student, retain_graph=True)[0]
    analytic = log_p.exp() * ((log_p - log_q) - kl[..., None]) * (mask / lengths[:, None] / 4)[..., None]
    torch.testing.assert_close(gradient, analytic, atol=1e-12, rtol=1e-10)
    results["sequence_weighted_logit_gradient_max_error"] = (gradient - analytic).abs().max().item()
    results["status"] = "passed"
    output = Path(__file__).with_name("verification.json")
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(output.read_text())


if __name__ == "__main__":
    main()
