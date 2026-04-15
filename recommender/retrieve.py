"""Candidate retrieval: cluster selection + KNN search within clusters.

Two-stage retrieval:
1. Find the nearest clusters to the user centroids (cheap, operates on k centroids).
   Budget is controlled by the diversity slider δ.
2. Brute-force KNN within those clusters, scoring each paper against all user
   centroids and taking the max similarity (nearest research thread).

All similarity computations use dot product (embeddings are unit-norm).
"""

from __future__ import annotations

import math

import numpy as np

from pipeline.index import PaperIndex


def find_nearest_clusters(
    user_centroids: np.ndarray,
    index_centroids: np.ndarray,
    diversity: float = 0.5,
) -> list[int]:
    """Find clusters to search, distributing budget across user centroids.

    The total cluster budget is ceil(2 + diversity * 3):
        δ=0.0 → 2 clusters, δ=0.5 → 4, δ=1.0 → 5.
    Budget is split evenly across the user's k_u centroids (at least 1 each).

    Args:
        user_centroids: Shape (k_u, 768), float32, unit-norm rows.
        index_centroids: Shape (k, 768), float32, unit-norm rows (k-means centroids).
        diversity: The δ slider value, 0.0–1.0.

    Returns:
        Deduplicated list of cluster indices to search.
    """
    total_budget = math.ceil(2 + diversity * 3)
    k_u = user_centroids.shape[0]
    per_centroid = max(1, total_budget // k_u)

    selected: set[int] = set()
    for u_i in user_centroids:
        sims = index_centroids @ u_i  # (k,)
        top = np.argsort(sims)[::-1][:per_centroid]
        selected.update(top.tolist())

    return list(selected)


def knn_in_clusters(
    user_centroids: np.ndarray,
    target_cluster_ids: list[int],
    index: PaperIndex,
    seen_ids: set[str],
    k: int = 40,
) -> list[tuple[float, dict, int]]:
    """Find the k most similar papers within the specified clusters.

    Each paper's similarity is the maximum dot product across all user
    centroids — i.e., scored against the user's closest research thread.

    Args:
        user_centroids: Shape (k_u, 768), float32, unit-norm rows.
        target_cluster_ids: Cluster IDs to search within.
        index: Loaded PaperIndex.
        seen_ids: Paper IDs to exclude.
        k: Max candidates to return.

    Returns:
        List of (max_similarity, paper_meta_dict, nearest_centroid_idx) tuples,
        sorted by descending similarity.
    """
    mask = np.isin(index.cluster_ids, target_cluster_ids)
    cand_indices = np.where(mask)[0]

    if len(cand_indices) == 0:
        return []

    cand_embs = index.embeddings[cand_indices]         # (M, 768)
    sim_matrix = cand_embs @ user_centroids.T           # (M, k_u)
    max_sims = sim_matrix.max(axis=1)                   # (M,)
    nearest_centroid = sim_matrix.argmax(axis=1)         # (M,)

    sorted_order = np.argsort(max_sims)[::-1]

    results: list[tuple[float, dict, int]] = []
    for idx in sorted_order:
        original_idx = cand_indices[idx]
        meta = index.paper_meta[original_idx]
        if meta["id"] in seen_ids:
            continue
        results.append((
            float(max_sims[idx]),
            meta,
            int(nearest_centroid[idx]),
        ))
        if len(results) >= k:
            break

    return results
