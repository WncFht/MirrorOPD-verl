# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU/FP64 checks for local OPD predictors and their counterexamples.

The nonlinear example has two fixed states and two actions. No GPU, model
checkpoint, or autoregressive rollout is involved. Spectral statements are
checked separately in the exact quadratic surrogate where they apply.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def probabilities(logits):
    p = 1 / (1 + np.exp(-logits))
    return np.stack([1 - p, p], axis=-1)


def kl(p, q):
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=-1)))


def same_divergence_check():
    scales = np.sqrt(np.array([8.0, 0.08]))
    fisher = scales**2 / 8
    p0 = probabilities(np.zeros(2))
    step_size, updates, amplitude = 0.1, 64, 0.1
    results = []
    for active in range(2):
        target_logits = np.zeros(2)
        target_logits[active] = amplitude
        teacher = probabilities(target_logits)
        theta = np.zeros(2)
        initial_loss = kl(p0, teacher)
        initial_gradient = scales * target_logits / 8
        path = []
        for step in range(updates + 1):
            logits = scales * theta
            p = probabilities(logits)
            loss = kl(p, teacher)
            path.append({"update": step, "rkl": loss})
            if step < updates:
                descent = scales * p[:, 1] * p[:, 0] * (target_logits - logits) / 2
                theta += step_size * descent
        log_ratio = np.log(teacher) - np.log(p0)
        centered = log_ratio - np.sum(p0 * log_ratio, axis=-1, keepdims=True)
        results.append(
            {
                "teacher": "fast-state" if active == 0 else "slow-state",
                "rkl": initial_loss,
                "fkl": kl(teacher, p0),
                "chi_square": float(np.mean(np.sum(p0 * (teacher / p0 - 1) ** 2, axis=-1))),
                "log_ratio_variance": float(np.mean(np.sum(p0 * centered**2, axis=-1))),
                "normalized_initial_gradient_energy": float(initial_gradient @ initial_gradient / (2 * initial_loss)),
                "mode_eigenvalue": float(fisher[active]),
                "remaining_fraction": path[-1]["rkl"] / initial_loss,
                "quadratic_remaining_prediction": float((1 - step_size * fisher[active]) ** (2 * updates)),
                "final_rkl": path[-1]["rkl"],
                "path": path,
            }
        )
    for metric in ["rkl", "fkl", "chi_square", "log_ratio_variance"]:
        assert abs(results[0][metric] - results[1][metric]) < 1e-14, metric
    return {"step_size": step_size, "updates": updates, "amplitude": amplitude, "results": results}


def spectral_and_source_checks():
    eigenvalues = np.array([1.0, 0.1, 0.01])
    operator = np.diag(eigenvalues)
    signal = np.array([0.2, -0.3, 0.4])
    alpha, steps = 0.2, 32
    residual = signal.copy()
    for _ in range(steps):
        residual -= alpha * operator @ residual
    expected = (1 - alpha * eigenvalues) ** steps * signal
    spectral_error = float(np.max(np.abs(residual - expected)))
    advantage = np.array([1.0, -2.0, 3.0])
    advantage /= np.linalg.norm(advantage)
    teacher_step = 0.05
    rl_signal = teacher_step * eigenvalues * advantage
    bounds = []
    for time in [0.5, 1.0, 10.0, 100.0, 1000.0]:
        loss = float(np.sum(rl_signal**2 * np.exp(-2 * eigenvalues * time)) / 2)
        bound = float(teacher_step**2 / (2 * np.e**2 * time**2))
        assert loss <= bound + 1e-15
        bounds.append({"time": time, "loss": loss, "absolute_source_bound": bound})
    slow_advantage = np.array([0.0, 0.0, 1.0])
    slow_signal = teacher_step * eigenvalues * slow_advantage
    initial_slow_loss = float(slow_signal @ slow_signal / 2)
    relative_slow_loss = float(np.exp(-2 * eigenvalues[-1] * 10))
    # In a two-dimensional Hilbert space, a task-orthogonal signal can be fast.
    task_signal = np.array([0.0, 1.0])
    teacher_signal = np.array([1.0, 0.0])
    jacobian = np.diag(np.sqrt([1.0, 0.01]))
    task_gradient = jacobian.T @ task_signal
    opd_gradient = jacobian.T @ teacher_signal
    checks = {
        "discrete_spectral_max_error": spectral_error,
        "rl_source_bounds": bounds,
        "slow_only_rl_initial_loss": initial_slow_loss,
        "slow_only_rl_remaining_at_t10": relative_slow_loss,
        "task_vs_opd_gradient_inner_product": float(task_gradient @ opd_gradient),
        "task_orthogonal_signal_remaining_at_t10": float(np.exp(-20)),
    }
    assert spectral_error < 1e-14
    return checks


def cross_distribution_check():
    # PSD training/evaluation curvatures can still induce negative transfer.
    train = np.array([[1.0, 0.0], [0.0, 0.0]])
    evaluation = np.array([[1.0, 2.0], [2.0, 4.0]])
    target = np.array([1.0, -1.0])
    train_direction = train @ target
    evaluation_direction = evaluation @ target
    initial = float(target @ evaluation @ target / 2)
    eta = 1e-4
    error_after = target - eta * train_direction
    final = float(error_after @ evaluation @ error_after / 2)
    measured = (initial - final) / eta
    first_order = float(train_direction @ evaluation_direction)
    assert first_order < 0
    assert abs(measured - first_order) < 1e-3
    return {"cross_gradient_prediction": first_order, "measured_gain_per_step_size": measured}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "settings": {"dtype": "float64", "device": "cpu", "autoregressive": False},
        "same_divergence": same_divergence_check(),
        "spectral_and_source": spectral_and_source_checks(),
        "cross_distribution": cross_distribution_check(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
    }
    (args.output / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    compact = [{k: v for k, v in row.items() if k != "path"} for row in result["same_divergence"]["results"]]
    print(json.dumps({"same_divergence": compact, "spectral_and_source": result["spectral_and_source"]}, indent=2))


if __name__ == "__main__":
    main()
