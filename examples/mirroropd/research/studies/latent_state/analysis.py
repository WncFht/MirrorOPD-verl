# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Fit a fixed training-only reference; score disjoint saved response queries."""

from pathlib import Path

import numpy as np

from examples.mirroropd.research.core.io import dump, read

from .artifacts import cached, digest, fingerprint, publish
from .extract import task_key
from .geometry import cosine_knn, coverage_draws, js_divergence, occupancy


def load_signatures(plan_path, pair):
    plan_path = Path(plan_path)
    plan = read(plan_path)
    shards, identities = [], []
    for task in plan["pairs"][pair]["tasks"]:
        path = plan_path.parent / "signatures" / pair / f"{task['id']}.npz"
        receipt = read(path.with_suffix(".json"))
        key = task_key(plan, pair, task, receipt["settings"])
        if not cached(path, key):
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            shards.append({k: archive[k] for k in archive.files})
        identities.append(receipt["sha256"])
    arrays = {k: np.concatenate([s[k] for s in shards]) for k in shards[0]}
    if not np.isfinite(arrays["features"]).all() or np.any(np.linalg.norm(arrays["features"], axis=1) == 0):
        raise ValueError("Nonfinite or zero signatures")
    return arrays, identities


def reference_masks(data):
    """Banks never influence fitting; all bank trajectories remain query eligible."""
    metric = data["action"] > 0
    training = ~np.isin(data["group"], ["initial", "teacher"])
    reference = np.flatnonzero(metric & training & (data["split"] == "reference"))
    test = metric & (~training | (data["split"] == "test"))
    return reference, test, training


def balanced_js(labels, data, test, groups, k, seed, draws=30):
    """Equal trajectory budgets against held-out training; target excludes query IDs."""
    training = ~np.isin(data["group"], ["initial", "teacher"])
    target_mask = test & training
    target_pool = np.unique(data["trajectory"][target_mask])
    pools = {g: np.unique(data["trajectory"][test & (data["group"] == g)]) for g in groups}
    pools["training_test"] = target_pool
    budget = min(128, len(target_pool) // 2, *(len(p) for p in pools.values()))
    if budget < 2:
        raise ValueError("Too few held-out training trajectories")
    rng = np.random.default_rng(seed)
    values = {g: [] for g in pools}
    for _ in range(draws):
        for g, pool in pools.items():
            chosen = rng.choice(pool, budget, replace=False)
            remaining = np.setdiff1d(target_pool, chosen)
            target = rng.choice(remaining, budget, replace=False)
            a = occupancy(labels[test & np.isin(data["trajectory"], chosen)], k)
            b = occupancy(labels[target_mask & np.isin(data["trajectory"], target)], k)
            values[g].append(js_divergence(a, b))
    return budget, {g: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), values=v) for g, v in values.items()}


def analyze(plan_path, pair, *, ks=(50, 100, 200), components=50):
    import sklearn
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    plan_path = Path(plan_path)
    plan = read(plan_path)
    data, identities = load_signatures(plan_path, pair)
    output = plan_path.parent / "analysis" / pair
    groups, seed = plan["groups"], plan["seed"]
    features, group = data["features"], data["group"]
    reference, test, training = reference_masks(data)
    if len(reference) < max(max(ks), components + 1):
        raise ValueError("Insufficient training reference")
    pca = PCA(n_components=components, svd_solver="full").fit(features[reference])
    reduced = pca.transform(features)
    distances = np.full(len(group), np.nan)
    distances[test] = cosine_knn(features[test], features[reference])
    p95 = float(np.quantile(distances[test & training], 0.95))
    masks = {g: test & (group == g) for g in groups}
    masks["training_test"] = test & training
    pools = {g: (mask, sorted(set(data["trajectory"][mask]))) for g, mask in masks.items()}
    budget = min(128, *(len(pool) for _, pool in pools.values()))
    if budget < 2:
        raise ValueError("Fewer than two query trajectories in a group")
    metrics, centers, label_sets = {}, {}, {}
    for k in ks:
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(reduced[reference])
        labels = model.predict(reduced)
        centers[f"centers_{k}"] = model.cluster_centers_
        label_sets[str(k)] = labels
        cover = coverage_draws(labels, data["trajectory"], pools, k, n_prompts=budget, seed=seed)
        js_budget, js = balanced_js(labels, data, test, groups, k, seed)
        target_occupancy = occupancy(labels[test & training], k)
        metrics[str(k)] = {}
        for g, mask in masks.items():
            share = occupancy(labels[mask], k)
            metrics[str(k)][g] = dict(
                states=int(mask.sum()),
                trajectories=len(pools[g][1]),
                occupancy=share.tolist(),
                occupancy_counts=np.bincount(labels[mask], minlength=k).tolist(),
                js_to_training=js_divergence(share, target_occupancy),
                balanced_js=js[g],
                coverage=cover[g],
                reference_knn_cosine=float(distances[mask].mean()),
                beyond_reference_p95=float((distances[mask] > p95).mean()),
            )
    # The map is exploratory and transductive; neither its coordinates nor t-SNE distances enter metrics.
    xy = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=seed).fit_transform(reduced)
    key = fingerprint(
        dict(
            sources=identities,
            ks=ks,
            components=components,
            code=digest(__file__),
            geometry=digest(Path(__file__).with_name("geometry.py")),
            numpy=np.__version__,
            sklearn=sklearn.__version__,
        )
    )
    publish(
        output / "geometry.npz",
        key,
        dict(
            reference_indices=reference,
            pca_mean=pca.mean_,
            pca_components=pca.components_,
            explained_variance_ratio=pca.explained_variance_ratio_,
            **centers,
        ),
        sources=identities,
    )
    result = dict(
        pair=pair,
        provenance=dict(
            plan_sha256=digest(plan_path),
            signature_sha256=identities,
            analysis_sha256=digest(__file__),
            geometry_sha256=digest(Path(__file__).with_name("geometry.py")),
            numpy=np.__version__,
            sklearn=sklearn.__version__,
            seed=seed,
        ),
        groups=groups,
        reference_states=len(reference),
        protocol="training-only-reference",
        reference_states_per_group={g: int(np.sum(group[reference] == g)) for g in groups},
        explained_variance=float(pca.explained_variance_ratio_.sum()),
        coverage_trajectory_budget=budget,
        js_trajectory_budget=js_budget,
        js_baseline="Disjoint held-out training subsets; all comparisons exclude query IDs from target draws",
        training_test_trajectories=len(pools["training_test"][1]),
        reference_knn_p95=p95,
        metrics=metrics,
        interpretation=(
            "Training-only reference; whole-prompt holdout; all bank trajectories query only; "
            "no cross-pair coordinate alignment"
        ),
    )
    points = []
    for i in range(len(group)):
        points.append(
            dict(
                x=float(xy[i, 0]),
                y=float(xy[i, 1]),
                group=str(group[i]),
                position=float(data["action"][i] / max(1, data["response_tokens"][i] - 1)),
                trajectory=str(data["trajectory"][i]),
                order=int(data["rank"][i]),
                metadata=dict(
                    outer=int(data["outer"][i]),
                    split="query" if group[i] in ("initial", "teacher") else str(data["split"][i]),
                    action=int(data["action"][i]),
                    response_tokens=int(data["response_tokens"][i]),
                    cluster=int(label_sets[str(ks[-1])][i]),
                    clusters={k: int(v[i]) for k, v in label_sets.items()},
                    prompt=str(data["prompt"][i]),
                    prompt_text=str(data.get("prompt_text", np.full(len(group), ""))[i]),
                    context=str(data.get("context", np.full(len(group), ""))[i]),
                ),
            )
        )
    dump(
        output / "result.json",
        dict(
            metrics=result,
            map=dict(points=points, projection=f"t-SNE of training-reference PCA{components}; all plotted states"),
        ),
    )
    return result
