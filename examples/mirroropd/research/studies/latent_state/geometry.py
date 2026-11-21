# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Teacher signatures and shared reference geometry, adapted from explcre latent_map."""

import numpy as np


def state_positions(prompt_length, response_length, mode="relative8"):
    if min(prompt_length, response_length) <= 0:
        raise ValueError("Prompt/response must be nonempty")
    if mode == "relative8":
        # Hidden index immediately BEFORE a selected response action; includes s_0.
        actions = np.unique(np.linspace(0, response_length - 1, 8).astype(int))
        return (prompt_length - 1 + actions).tolist()
    if mode == "final":
        # Explicitly different convention: AFTER the final nonpadding response token.
        return [prompt_length + response_length - 1]
    if mode in {"absolute2000", "absolute4000"}:
        index = int(mode.removeprefix("absolute")) - 1  # one-based absolute position in prompt+response
        return [index] if index < prompt_length + response_length else []
    raise ValueError(mode)


def balanced_indices(bank, names, seed=42):
    """Equal number of states per bank (the smallest bank's size), sampled without replacement, seeded."""
    rng = np.random.default_rng(seed)
    pools = [np.flatnonzero(bank == b) for b in names]
    n = min(len(p) for p in pools)
    return np.concatenate([np.sort(rng.choice(p, n, replace=False)) for p in pools])


def occupancy(labels, k):
    """Share of states in each of the k clusters."""
    return np.bincount(labels, minlength=k) / max(len(labels), 1)


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits (0 = identical, 1 = disjoint support)."""
    p, q = np.asarray(p, float) / np.sum(p), np.asarray(q, float) / np.sum(q)
    m = (p + q) / 2
    kl = lambda x: float(np.sum(x[x > 0] * np.log2(x[x > 0] / m[x > 0])))
    return (kl(p) + kl(q)) / 2


def coverage(labels, k):
    """Cov(S): fraction of the k reference clusters reached by the states' labels."""
    return np.unique(labels).size / k


def coverage_draws(labels, prompt, groups, k, n_prompts=32, n_draws=30, seed=42):
    """Equal-budget coverage: per draw, n_prompts prompts from each group's pool (groups with the same pool share
    the draw); groups = {name: (state mask, pool of prompt indices)}. Returns {name: {mean, sd, values}}."""
    rng = np.random.default_rng(seed)
    values = {g: [] for g in groups}
    for _ in range(n_draws):
        chosen = {}
        for g, (mask, pool) in groups.items():
            key = tuple(pool)
            if key not in chosen:
                chosen[key] = rng.choice(np.asarray(pool), n_prompts, replace=False)
            values[g].append(coverage(labels[mask & np.isin(prompt, chosen[key])], k))
    return {
        g: {
            "mean": float(np.mean(v)),
            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "values": [float(x) for x in v],
        }
        for g, v in values.items()
    }


def cosine_knn(query, reference, k=10, chunk=256):
    """Exact original-space mean kNN cosine distance, with bounded query working memory."""
    if len(reference) < k:
        raise ValueError("Too few reference states for kNN")
    reference = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    values = []
    for start in range(0, len(query), chunk):
        q = query[start : start + chunk]
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        distances = np.maximum(0, 1 - q @ reference.T)
        values.append(np.partition(distances, k - 1, axis=1)[:, :k].mean(axis=1))
    return np.concatenate(values)
