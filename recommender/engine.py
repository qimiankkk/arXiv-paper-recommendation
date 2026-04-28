"""Top-level recommendation engine.

Provides a single recommend() entry point that orchestrates retrieval and
reranking to produce the final list of recommended papers.
"""

from __future__ import annotations

import numpy as np

from pipeline.index import PaperIndex
from recommender.retrieve import find_nearest_clusters, knn_in_clusters_balanced
from recommender.rerank import rerank_and_select


def recommend(
    user_centroids: np.ndarray,
    seen_ids: set[str],
    index: PaperIndex,
    diversity: float = 0.5,
    n: int = 5,
    centroid_like_counts: list[int] | None = None,
) -> list[dict]:
    """Generate n paper recommendations for a user.

    Args:
        user_centroids: Shape (k_u, 768), float32, unit-norm rows.
        seen_ids: Set of arXiv paper IDs already seen.
        index: Loaded PaperIndex.
        diversity: δ slider value, 0.0–1.0.
        n: Papers to recommend. Default 5.
        centroid_like_counts: Optional like-count per user centroid index.

    Returns:
        List of up to n paper_meta dicts with "rec_score" added.
    """
    k_u = user_centroids.shape[0]

    # 1-2. Retrieval (balanced candidate pool across centroids)
    if diversity <= 0.0:
        # Most focused mode: exact full-corpus search with centroid-balanced quotas.
        all_clusters = list(range(index.centroids.shape[0]))
        candidates = knn_in_clusters_balanced(
            user_centroids=user_centroids,
            target_cluster_ids=all_clusters,
            index=index,
            seen_ids=seen_ids,
            k=40,
        )
    else:
        # Cluster-preselected retrieval for broader exploration modes.
        clusters = find_nearest_clusters(user_centroids, index.centroids, diversity)
        candidates = knn_in_clusters_balanced(
            user_centroids,
            clusters,
            index,
            seen_ids,
            k=40,
        )

    # 3. Rerank + diversity filter
    results = rerank_and_select(
        candidates,
        k_u=k_u,
        diversity=diversity,
        n=n,
        centroid_like_counts=centroid_like_counts,
    )

    # Fallback: if too few results, search all clusters
    if len(results) < n:
        all_clusters = list(range(index.centroids.shape[0]))
        selected_ids = {r["id"] for r in results}
        expanded_seen = seen_ids | selected_ids
        all_candidates = knn_in_clusters_balanced(
            user_centroids, all_clusters, index, expanded_seen, k=40
        )
        extra = rerank_and_select(
            all_candidates,
            k_u=k_u,
            diversity=diversity,
            n=n - len(results),
            centroid_like_counts=centroid_like_counts,
        )
        results.extend(extra)

    return results
