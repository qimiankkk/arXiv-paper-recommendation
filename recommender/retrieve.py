"""Candidate retrieval: cluster selection + KNN search within clusters.

Two-stage retrieval:
1. Find the nearest clusters to the user embedding (cheap, operates on k centroids).
2. Brute-force KNN within those clusters (more expensive, but scoped to a subset).

All similarity computations use dot product (embeddings are unit-norm).
"""

from __future__ import annotations

import numpy as np

from pipeline.index import PaperIndex


def find_nearest_clusters(
    user_emb: np.ndarray,
    centroids: np.ndarray,
    n: int = 2,
) -> list[int]:
    """Find the n cluster centroids most similar to the user embedding.

    Args:
        user_emb: Shape (768,), float32, unit-norm.
        centroids: Shape (k, 768), float32, unit-norm rows.
        n: Number of top clusters to return.

    Returns:
        List of n cluster indices (ints), sorted by descending similarity.
    """
    sims = centroids @ user_emb  # shape (k,)
    top_indices = np.argsort(sims)[::-1][:n]
    return top_indices.tolist()


def knn_in_clusters(
    user_emb: np.ndarray,
    target_cluster_ids: list[int],
    index: PaperIndex,
    seen_ids: set[str],
    k: int = 40,
) -> list[tuple[float, dict]]:
    """Find the k most similar papers within the specified clusters.

    Filters out papers the user has already seen.

    Args:
        user_emb: Shape (768,), float32, unit-norm.
        target_cluster_ids: List of cluster IDs to search within.
        index: The loaded PaperIndex containing all embeddings and metadata.
        seen_ids: Set of arXiv paper IDs to exclude (already seen by user).
        k: Maximum number of candidates to return.

    Returns:
        List of (similarity_score, paper_meta_dict) tuples, sorted by
        descending similarity. Length is min(k, available unseen papers).
    """
    mask = np.isin(index.cluster_ids, target_cluster_ids)
    cand_indices = np.where(mask)[0]

    if len(cand_indices) == 0:
        return []

    cand_embs = index.embeddings[cand_indices]  # shape (M, 768)
    sims = cand_embs @ user_emb  # shape (M,)

    # Sort by descending similarity
    sorted_order = np.argsort(sims)[::-1]

    results: list[tuple[float, dict]] = []
    for idx in sorted_order:
        original_idx = cand_indices[idx]
        meta = index.paper_meta[original_idx]
        if meta["id"] in seen_ids:
            continue
        results.append((float(sims[idx]), meta))
        if len(results) >= k:
            break

    return results
