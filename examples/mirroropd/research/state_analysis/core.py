# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""NumPy primitives for a shared PCA/K-means state-coverage space."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import numpy as np


def relative_positions(valid_positions: list[int], count: int) -> list[int]:
    """Choose up to ``count`` unique target positions across a response."""
    if count < 1:
        raise ValueError("positions-per-rollout must be positive")
    if not valid_positions:
        return []
    if len(valid_positions) <= count:
        return list(valid_positions)
    ranks = np.linspace(0, len(valid_positions) - 1, count, dtype=np.int64)
    return [valid_positions[int(rank)] for rank in ranks]


def fit_pca(values: np.ndarray, n_components: int) -> dict[str, np.ndarray | float]:
    """Fit centered PCA with a full SVD and return a serializable transform."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("PCA needs a 2D matrix with at least two rows and one feature")
    if not 1 <= n_components <= min(values.shape):
        raise ValueError(f"PCA dimensions must be in [1, {min(values.shape)}]")
    mean = values.mean(axis=0)
    centered = values - mean
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    variances = singular_values**2 / (values.shape[0] - 1)
    total = float(variances.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("PCA reference vectors have zero or non-finite variance")
    ratios = variances / total
    return {
        "mean": mean.astype(np.float32),
        "components": components[:n_components].astype(np.float32),
        "explained_variance": variances[:n_components].astype(np.float32),
        "explained_variance_ratio": ratios[:n_components].astype(np.float32),
        "total_explained_variance_ratio": float(ratios[:n_components].sum()),
    }


def transform_pca(values: np.ndarray, pca: dict[str, np.ndarray | float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - np.asarray(pca["mean"], dtype=np.float32)) @ np.asarray(pca["components"], dtype=np.float32).T


def squared_distances(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Compute Euclidean squared distances without a 3D intermediate."""
    values = np.asarray(values, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    distances = (
        np.sum(values * values, axis=1, keepdims=True)
        + np.sum(centers * centers, axis=1)[None, :]
        - 2.0 * values @ centers.T
    )
    np.maximum(distances, 0.0, out=distances)
    return distances


def assign_clusters(values: np.ndarray, centers: np.ndarray, *, chunk_size: int = 4096) -> np.ndarray:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    labels = np.empty(len(values), dtype=np.int32)
    for start in range(0, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        labels[start:stop] = squared_distances(values[start:stop], centers).argmin(axis=1)
    return labels


def _kmeans_plus_plus(values: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    n_samples = len(values)
    centers = np.empty((n_clusters, values.shape[1]), dtype=np.float32)
    first = int(rng.integers(n_samples))
    centers[0] = values[first]
    closest = squared_distances(values, centers[:1]).ravel()
    for index in range(1, n_clusters):
        total = float(closest.sum())
        if total <= 0 or not math.isfinite(total):
            choice = int(rng.integers(n_samples))
        else:
            choice = int(rng.choice(n_samples, p=closest / total))
        centers[index] = values[choice]
        candidate = squared_distances(values, centers[index : index + 1]).ravel()
        np.minimum(closest, candidate, out=closest)
    return centers


def fit_kmeans(
    values: np.ndarray,
    n_clusters: int,
    *,
    seed: int = 17,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> tuple[np.ndarray, dict[str, int | float | bool]]:
    """Fit Lloyd K-means with k-means++ initialization and deterministic RNG."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError("K-means needs a non-empty 2D matrix")
    if not 1 <= n_clusters <= len(values):
        raise ValueError(f"K must be in [1, {len(values)}]")
    if max_iter < 1 or tol < 0:
        raise ValueError("max_iter must be positive and tol nonnegative")

    centers = _kmeans_plus_plus(values, n_clusters, np.random.default_rng(seed))
    labels = np.full(len(values), -1, dtype=np.int32)
    converged = False
    for iteration in range(1, max_iter + 1):
        distances = squared_distances(values, centers)
        new_labels = distances.argmin(axis=1).astype(np.int32)
        min_distances = distances[np.arange(len(values)), new_labels]
        updated = np.empty_like(centers)
        for cluster in range(n_clusters):
            members = values[new_labels == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
            else:
                # Re-seed an empty center at a high-error point. The following
                # assignment pass determines whether the point forms a cluster.
                updated[cluster] = values[int(np.argmax(min_distances))]
        shift = float(np.sqrt(np.sum((updated - centers) ** 2, axis=1)).max())
        stable = np.array_equal(new_labels, labels)
        centers = updated
        labels = new_labels
        if stable or shift <= tol:
            converged = True
            break

    final_distances = squared_distances(values, centers)
    final_labels = final_distances.argmin(axis=1).astype(np.int32)
    occupied = int(np.unique(final_labels).size)
    if occupied != n_clusters:
        raise ValueError(
            f"K-means occupied {occupied}/{n_clusters} clusters; reduce K or check duplicate reference states"
        )
    info: dict[str, int | float | bool] = {
        "iterations": iteration,
        "converged": converged,
        "inertia": float(final_distances[np.arange(len(values)), final_labels].sum()),
        "occupied_clusters": occupied,
    }
    return centers, info


def deterministic_rollout_split(
    records: list[dict], *, source: str, group_field: str, seed: int, reference_fraction: float = 0.6
) -> tuple[set[str], set[str]]:
    """Split whole reference-source rollouts within each explicit grouping field."""
    if not 0 < reference_fraction < 1:
        raise ValueError("reference_fraction must be between zero and one")
    by_group: dict[str, set[str]] = defaultdict(set)
    for row in records:
        if row.get("source") != source:
            continue
        if group_field not in row or row[group_field] is None:
            raise ValueError(f"State record lacks requested split field {group_field!r}")
        by_group[str(row[group_field])].add(str(row["trajectory_id"]))
    if not by_group:
        raise ValueError(f"No states found for reference source {source!r}")

    reference: set[str] = set()
    heldout: set[str] = set()
    for group, rollout_ids in sorted(by_group.items()):
        ordered = sorted(
            rollout_ids,
            key=lambda rollout_id: hashlib.sha256(f"{seed}\0{group}\0{rollout_id}".encode()).hexdigest(),
        )
        if len(ordered) < 2:
            raise ValueError(f"Split group {group!r} has fewer than two rollouts")
        n_reference = min(len(ordered) - 1, max(1, int(round(reference_fraction * len(ordered)))))
        reference.update(ordered[:n_reference])
        heldout.update(ordered[n_reference:])
    return reference, heldout


def coverage(labels: np.ndarray, n_clusters: int) -> tuple[int, float]:
    visited = int(np.unique(labels).size)
    return visited, visited / n_clusters
