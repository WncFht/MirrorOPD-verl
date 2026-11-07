# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy<2"]
# ///
"""Exact CPU checks for source distributions and fresh-rollout RKL advantages.

This enumerates a finite prefix tree; it does not load models or launch training.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def softmax(logits):
    values = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


def normalize(values):
    return values / values.sum()


def kl(p, q):
    return float(np.dot(p, np.log(p / q)))


def chi2(p, q):
    return float(np.sum(p * p / q) - 1)


def geometric(p, q, teacher_weight):
    return softmax((1 - teacher_weight) * np.log(p) + teacher_weight * np.log(q))


def root_in_unit_interval(function):
    lo, hi = 0.0, 1.0
    if not function(lo) <= 1e-12 or not function(hi) >= -1e-12:
        raise ValueError("Expected a bracketed root.")
    for _ in range(70):
        mid = (lo + hi) / 2
        if function(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def center(p, teacher, family):
    if family == "geometric_mean":
        return geometric(p, teacher, 0.5)
    if family == "geometric_weighted":
        return geometric(p, teacher, 0.3)
    if family in ("reverse_minimax", "trust_region", "lp2", "smooth_max"):

        def residual(eta):
            q = geometric(p, teacher, eta)
            a, b = kl(q, p), kl(q, teacher)
            if family == "reverse_minimax":
                return a - b
            if family == "trust_region":
                return a - 0.3 * kl(teacher, p)
            if family == "lp2":
                return eta - b / (a + b)
            return eta - 1 / (1 + np.exp(-(b - a) / 0.2))

        eta = root_in_unit_interval(residual)
        return geometric(p, teacher, eta)
    if family in ("arithmetic_mean", "tv_midpoint"):
        return (p + teacher) / 2
    if family == "forward_minimax":

        def residual(eta):
            q = (1 - eta) * p + eta * teacher
            return kl(p, q) - kl(teacher, q)

        eta = root_in_unit_interval(residual)
        return (1 - eta) * p + eta * teacher
    if family == "hellinger":
        return normalize((np.sqrt(p) + np.sqrt(teacher)) ** 2)
    if family == "chi2_minimax":

        def residual(eta):
            q = normalize(np.sqrt((1 - eta) * p * p + eta * teacher * teacher))
            return chi2(p, q) - chi2(teacher, q)

        eta = root_in_unit_interval(residual)
        return normalize(np.sqrt((1 - eta) * p * p + eta * teacher * teacher))
    raise ValueError(f"Unknown family: {family}")


def prefix_probability(policy, prefix, indices):
    result = 1.0
    for step, action in enumerate(prefix):
        result *= policy[indices[prefix[:step]], action]
    return result


def sequence_probabilities(policy, paths, indices):
    return np.array([prefix_probability(policy, path, indices) for path in paths])


def marginal_masses(probabilities, paths, prefixes):
    return np.array(
        [sum(prob for prob, path in zip(probabilities, paths, strict=True) if path[: len(h)] == h) for h in prefixes]
    )


def local_negative_gradient(learner, teacher):
    log_ratio = np.log(learner / teacher)
    loss = np.sum(learner * log_ratio, axis=-1, keepdims=True)
    return -learner * (log_ratio - loss)


def sampled_gradient(masses, proposal, learner, teacher, weights, correct_action=True):
    result = np.zeros_like(learner)
    for index in range(len(masses)):
        for action in range(learner.shape[1]):
            action_ratio = learner[index, action] / proposal[index, action] if correct_action else 1.0
            advantage = weights[index] * action_ratio * np.log(teacher[index, action] / learner[index, action])
            score = -learner[index].copy()
            score[action] += 1
            result[index] += masses[index] * proposal[index, action] * advantage * score
    return result


def finite_difference_gradient(logits, teacher, source_masses):
    epsilon = 1e-5
    gradient = np.zeros_like(logits)

    def loss(candidate):
        p = softmax(candidate)
        return float(np.dot(source_masses, np.sum(p * np.log(p / teacher), axis=-1)))

    for coordinate in np.ndindex(logits.shape):
        plus, minus = logits.copy(), logits.copy()
        plus[coordinate] += epsilon
        minus[coordinate] -= epsilon
        gradient[coordinate] = -(loss(plus) - loss(minus)) / (2 * epsilon)
    return gradient


def max_error(left, right):
    return float(np.max(np.abs(left - right)))


def run_audit(seed):
    rng = np.random.default_rng(seed)
    horizon = 3
    prefixes = [h for length in range(horizon) for h in itertools.product(range(2), repeat=length)]
    paths = list(itertools.product(range(2), repeat=horizon))
    indices = {prefix: index for index, prefix in enumerate(prefixes)}
    logits = rng.normal(size=(len(prefixes), 2))
    mu = softmax(logits)
    initial = softmax(rng.normal(size=logits.shape))
    teacher = softmax(rng.normal(size=logits.shape))
    moved_logits = logits + 0.6 * rng.normal(size=logits.shape)
    moved = softmax(moved_logits)
    policies = {"current": mu, "initial": initial, "teacher": teacher}
    families = (
        "geometric_mean",
        "geometric_weighted",
        "reverse_minimax",
        "trust_region",
        "lp2",
        "smooth_max",
        "arithmetic_mean",
        "forward_minimax",
        "hellinger",
        "chi2_minimax",
        "tv_midpoint",
    )
    for family in families:
        policies[family] = np.array([center(p, t, family) for p, t in zip(mu, teacher, strict=True)])
    sequences = {name: sequence_probabilities(policy, paths, indices) for name, policy in policies.items()}
    sequences["trajectory_mixture"] = (sequences["current"] + sequences["teacher"]) / 2
    sequences["sequence_geometric"] = normalize(np.sqrt(sequences["current"] * sequences["teacher"]))
    masses = {name: marginal_masses(prob, paths, prefixes) for name, prob in sequences.items()}
    fresh_masses = masses["current"]
    case_results = {}
    for name, source_masses in masses.items():
        weights = source_masses / fresh_masses
        direct = source_masses[:, None] * local_negative_gradient(mu, teacher)
        equivalent = sampled_gradient(fresh_masses, mu, mu, teacher, weights)
        reused_direct = source_masses[:, None] * local_negative_gradient(moved, teacher)
        reused = sampled_gradient(fresh_masses, mu, moved, teacher, weights)
        uncorrected = sampled_gradient(fresh_masses, mu, moved, teacher, weights, correct_action=False)
        normalization_errors = [
            abs(sum(fresh_masses[i] * weights[i] for i, h in enumerate(prefixes) if len(h) == t) - 1)
            for t in range(horizon)
        ]
        prefix_product_error = 0.0
        sampled_source_error = 0.0
        if name in policies:
            q = policies[name]
            product_weights = np.array(
                [prefix_probability(q, h, indices) / prefix_probability(mu, h, indices) for h in prefixes]
            )
            prefix_product_error = max_error(weights, product_weights)
            sampled_source = sampled_gradient(source_masses, q, moved, teacher, np.ones(len(prefixes)))
            sampled_source_error = max_error(reused_direct, sampled_source)
        conditional_weights = np.array(
            [
                sum(
                    sequences["current"][j] / fresh_masses[i] * sequences[name][j] / sequences["current"][j]
                    for j, path in enumerate(paths)
                    if path[: len(h)] == h
                )
                for i, h in enumerate(prefixes)
            ]
        )
        case_results[name] = {
            "fresh_gradient_error": max_error(direct, equivalent),
            "reused_gradient_error": max_error(reused_direct, reused),
            "finite_difference_error": max_error(
                reused_direct, finite_difference_gradient(moved_logits, teacher, source_masses)
            ),
            "weight_normalization_error": float(max(normalization_errors)),
            "prefix_product_error": prefix_product_error,
            "sequence_to_prefix_conditional_expectation_error": max_error(weights, conditional_weights),
            "source_action_correction_error": sampled_source_error,
            "missing_reuse_action_ratio_gradient_error": max_error(reused_direct, uncorrected),
        }

    prefix_ratio = masses["teacher"] / fresh_masses
    raw_weight = np.sqrt(prefix_ratio)
    paper_weight = raw_weight.copy()
    for length in range(horizon):
        mask = np.array([len(h) == length for h in prefixes])
        paper_weight[mask] /= np.dot(fresh_masses[mask], raw_weight[mask])
    local_gradient = masses["geometric_mean"][:, None] * local_negative_gradient(mu, teacher)
    incorrect_paper = sampled_gradient(fresh_masses, mu, mu, teacher, paper_weight)
    incorrect_raw = sampled_gradient(fresh_masses, mu, mu, teacher, raw_weight)
    paper_masses = fresh_masses * paper_weight
    inconsistency = max(
        abs(sum(paper_masses[indices[h + (a,)]] for a in range(2)) - paper_masses[indices[h]])
        for h in prefixes
        if len(h) < horizon - 1
    )
    counterexamples = {
        "paper_prefix_vs_local_gradient_error": max_error(local_gradient, incorrect_paper),
        "missing_local_normalizers_gradient_error": max_error(local_gradient, incorrect_raw),
        "paper_prefix_marginal_inconsistency": float(inconsistency),
        "sequence_vs_local_probability_difference": max_error(
            sequences["sequence_geometric"], sequences["geometric_mean"]
        ),
        "trajectory_vs_token_mixture_probability_difference": max_error(
            sequences["trajectory_mixture"], sequences["arithmetic_mean"]
        ),
    }

    # Answer-mean reductions involve future lengths, even for prefix-local KL.
    lengths = np.array([path.index(1) + 1 if 1 in path else horizon for path in paths], dtype=float)
    inverse_lengths = 1 / lengths
    active = np.array([1 not in h for h in prefixes], dtype=float)

    def length_weighted_masses(probabilities):
        return marginal_masses(probabilities * inverse_lengths, paths, prefixes) * active

    initial_answer_masses = length_weighted_masses(sequences["initial"])
    fresh_answer_masses = length_weighted_masses(sequences["current"])
    initial_prefix_weights = masses["initial"] / fresh_masses
    correct_answer_gradient = initial_answer_masses[:, None] * local_negative_gradient(mu, teacher)
    incorrect_answer_gradient = (fresh_answer_masses * initial_prefix_weights)[:, None] * local_negative_gradient(
        mu, teacher
    )
    counterexamples["fresh_length_weight_with_prefix_is_gradient_error"] = max_error(
        correct_answer_gradient, incorrect_answer_gradient
    )

    # Enumerate independent complete answers to distinguish exact IS from SNIS.
    proposal = sequences["current"]
    target = sequences["sequence_geometric"]
    unnormalized = np.sqrt(sequences["teacher"] / proposal)
    partition = float(np.dot(proposal, unnormalized))
    observable = np.array([path[0] for path in paths], dtype=float)
    target_mean = float(np.dot(target, observable))
    known_partition_mean = float(np.dot(proposal, unnormalized / partition * observable))
    single_sample_weights = unnormalized / unnormalized
    single_sample_mean = float(np.dot(proposal, single_sample_weights * observable))
    two_sample_mean = sum(
        proposal[i]
        * proposal[j]
        * (unnormalized[i] * observable[i] + unnormalized[j] * observable[j])
        / (unnormalized[i] + unnormalized[j])
        for i in range(len(paths))
        for j in range(len(paths))
    )
    normalization_checks = {
        "partition": partition,
        "target_mean": target_mean,
        "known_partition_single_sample_expectation_error": abs(known_partition_mean - target_mean),
        "snis_single_sample_weight_error": max_error(single_sample_weights, np.ones(len(paths))),
        "snis_single_sample_expectation": single_sample_mean,
        "proposal_mean": float(np.dot(proposal, observable)),
        "snis_single_sample_bias": abs(single_sample_mean - target_mean),
        "snis_two_sample_expectation": float(two_sample_mean),
        "snis_two_sample_bias": abs(float(two_sample_mean) - target_mean),
    }
    if normalization_checks["known_partition_single_sample_expectation_error"] > 1e-12:
        raise AssertionError("Known-partition IS should be unbiased even with one sample.")
    if normalization_checks["snis_single_sample_weight_error"] > 1e-12:
        raise AssertionError("A single SNIS sample should have unit normalized weight.")
    if min(normalization_checks[f"snis_{n}_sample_bias"] for n in ("single", "two")) < 1e-5:
        raise AssertionError("Expected finite-sample SNIS bias for this example.")

    # The same finite-difference check applies to other local learner losses.
    other_losses = {}
    for name in ("forward_kl", "hellinger"):
        ratio = teacher / moved
        if name == "forward_kl":
            advantage = ratio
            analytic = teacher - moved
        else:
            advantage = 0.5 * np.sqrt(ratio)
            overlap = np.sqrt(moved * teacher)
            analytic = 0.5 * (overlap - moved * overlap.sum(axis=-1, keepdims=True))
        sampled = moved * (advantage - (moved * advantage).sum(axis=-1, keepdims=True))
        other_losses[name] = max_error(analytic, sampled)

    verified_fields = [name for name in next(iter(case_results.values())) if not name.startswith("missing_")]
    largest_error = max(case[field] for case in case_results.values() for field in verified_fields)
    if largest_error > 2e-8 or max(other_losses.values()) > 1e-12:
        raise AssertionError(f"Identity check failed: {largest_error}")
    if min(counterexamples.values()) < 1e-5:
        raise AssertionError("Expected a non-degenerate counterexample.")
    if min(case["missing_reuse_action_ratio_gradient_error"] for case in case_results.values()) < 1e-5:
        raise AssertionError("Expected action correction to matter after a learner update.")
    return {
        "seed": seed,
        "horizon": horizon,
        "complete_paths": len(paths),
        "sources": case_results,
        "max_identity_error": largest_error,
        "other_learner_loss_errors": other_losses,
        "counterexamples": counterexamples,
        "normalization_checks": normalization_checks,
        "scope": "CPU FP64 exact finite-tree enumeration; fixed sources; no model training",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()
    result = run_audit(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
