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


def _is_withdrawn_paper(meta: dict) -> bool:
    """Return True when metadata indicates a withdrawn paper.

    arXiv withdrawal signals are usually present in title/abstract text
    (e.g. "This paper has been withdrawn by the author(s)").
    """
    text = f"{meta.get('title', '')} {meta.get('abstract', '')}".lower()
    return any(
        marker in text
        for marker in (
            "withdrawn",
            "retracted",
            "withdrawal",
            "this paper has been withdrawn",
        )
    )


def _compute_centroid_quotas(
    k_u: int,
    k: int,
    centroid_like_counts: list[int] | None = None,
) -> list[int]:
    """Compute candidate quotas per centroid.

    Rules:
    - If like counts exist and any > 0: allocate proportionally to likes.
    - Otherwise: near-even split.
    - If k >= k_u: try to keep at least one slot per centroid.
    """
    if k_u <= 0:
        return []
    if k <= 0:
        return [0 for _ in range(k_u)]

    has_like_signal = (
        centroid_like_counts is not None
        and len(centroid_like_counts) == k_u
        and any(c > 0 for c in centroid_like_counts)
    )

    if not has_like_signal:
        base = k // k_u
        rem = k % k_u
        return [base + (1 if i < rem else 0) for i in range(k_u)]

    likes = [max(0, int(x)) for x in centroid_like_counts]  # type: ignore[arg-type]
    if k < k_u:
        order = sorted(range(k_u), key=lambda i: likes[i], reverse=True)
        quotas = [0 for _ in range(k_u)]
        for i in order[:k]:
            quotas[i] = 1
        return quotas

    quotas = [1 for _ in range(k_u)]
    remaining = k - k_u
    total = sum(likes)
    if remaining <= 0 or total <= 0:
        return quotas

    raw = [(remaining * w) / total for w in likes]
    extra = [int(x) for x in raw]
    quotas = [q + e for q, e in zip(quotas, extra)]
    left = remaining - sum(extra)
    if left > 0:
        order = sorted(
            range(k_u),
            key=lambda i: (raw[i] - extra[i], likes[i], -i),
            reverse=True,
        )
        for i in order[:left]:
            quotas[i] += 1
    return quotas


def find_nearest_clusters(
    user_centroids: np.ndarray,
    index_centroids: np.ndarray,
    diversity: float = 0.5,
) -> list[int]:
    """Find clusters to search, distributing budget across user centroids.

    Note:
        In the serving engine, δ=0 is handled by global retrieval and bypasses
        this function. When this function is called, the cluster budget is
        ceil(2 + diversity * 3) (e.g. δ=0.5 -> 4, δ=1.0 -> 5).
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
        if _is_withdrawn_paper(meta):
            continue
        results.append((
            float(max_sims[idx]),
            meta,
            int(nearest_centroid[idx]),
        ))
        if len(results) >= k:
            break

    return results


def knn_global(
    user_centroids: np.ndarray,
    index: PaperIndex,
    seen_ids: set[str],
    k: int = 40,
) -> list[tuple[float, dict, int]]:
    """Find k most similar papers via full-corpus brute-force search.

    This bypasses cluster preselection and scores all papers directly.
    Used for the most focused mode (δ=0) where exact global similarity
    is preferred over approximate cluster-restricted retrieval.
    """
    all_clusters = list(range(index.centroids.shape[0]))
    return knn_in_clusters(
        user_centroids=user_centroids,
        target_cluster_ids=all_clusters,
        index=index,
        seen_ids=seen_ids,
        k=k,
    )


def knn_in_clusters_balanced(
    user_centroids: np.ndarray,
    target_cluster_ids: list[int],
    index: PaperIndex,
    seen_ids: set[str],
    k: int = 40,
    centroid_like_counts: list[int] | None = None,
) -> list[tuple[float, dict, int]]:
    """Retrieve candidates with per-centroid quotas to avoid pool domination.

    The function first reserves quota slots per centroid and pulls papers whose
    nearest centroid equals that centroid. It then backfills remaining slots by
    global max similarity.
    """
    mask = np.isin(index.cluster_ids, target_cluster_ids)
    cand_indices = np.where(mask)[0]
    if len(cand_indices) == 0 or k <= 0:
        return []

    cand_embs = index.embeddings[cand_indices]
    sim_matrix = cand_embs @ user_centroids.T
    max_sims = sim_matrix.max(axis=1)
    nearest_centroid = sim_matrix.argmax(axis=1)
    k_u = user_centroids.shape[0]
    quotas = _compute_centroid_quotas(k_u, k, centroid_like_counts)

    selected_local: list[int] = []
    selected_global_ids: set[str] = set()

    # Pass 1: fill reserved quota from papers nearest to each centroid.
    for ci in range(k_u):
        need = quotas[ci] if ci < len(quotas) else 0
        if need <= 0:
            continue
        local_ids = np.where(nearest_centroid == ci)[0]
        if len(local_ids) == 0:
            continue
        ordered = local_ids[np.argsort(max_sims[local_ids])[::-1]]
        picked = 0
        for local_i in ordered:
            meta = index.paper_meta[cand_indices[local_i]]
            pid = meta["id"]
            if pid in seen_ids or pid in selected_global_ids:
                continue
            if _is_withdrawn_paper(meta):
                continue
            selected_local.append(int(local_i))
            selected_global_ids.add(pid)
            picked += 1
            if picked >= need:
                break

    # Pass 2: backfill to k by global max similarity.
    if len(selected_local) < k:
        global_order = np.argsort(max_sims)[::-1]
        for local_i in global_order:
            if len(selected_local) >= k:
                break
            meta = index.paper_meta[cand_indices[local_i]]
            pid = meta["id"]
            if pid in seen_ids or pid in selected_global_ids:
                continue
            if _is_withdrawn_paper(meta):
                continue
            selected_local.append(int(local_i))
            selected_global_ids.add(pid)

    results: list[tuple[float, dict, int]] = []
    for local_i in selected_local:
        orig_i = cand_indices[local_i]
        meta = index.paper_meta[orig_i]
        results.append(
            (float(max_sims[local_i]), meta, int(nearest_centroid[local_i]))
        )
    return results
