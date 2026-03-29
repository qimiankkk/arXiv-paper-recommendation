"""Top-level recommendation engine.

Provides a single recommend() entry point that orchestrates retrieval and
reranking to produce the final list of recommended papers.
"""

from __future__ import annotations

import numpy as np

from pipeline.index import PaperIndex
from recommender.retrieve import find_nearest_clusters, knn_in_clusters
from recommender.rerank import rerank_and_select


def recommend(
    user_emb: np.ndarray,
    seen_ids: set[str],
    index: PaperIndex,
    n: int = 3,
) -> list[dict]:
    """Generate n paper recommendations for a user.

    Orchestrates the full recommendation pipeline:
    1. Find nearest clusters to the user embedding.
    2. KNN search within those clusters (excluding seen papers).
    3. Rerank with recency boost and diversity filter.

    Args:
        user_emb: Shape (768,), float32, unit-norm user embedding.
        seen_ids: Set of arXiv paper IDs the user has already seen.
        index: The loaded PaperIndex containing all embeddings and metadata.
        n: Number of papers to recommend. Default 3.

    Returns:
        List of up to n paper_meta dicts, each with an added "rec_score" key.
        May return fewer than n if the user has seen most papers or the
        index is small.
    """
    # 1. Find nearest clusters
    clusters = find_nearest_clusters(user_emb, index.centroids, n=2)

    # 2. KNN within those clusters
    candidates = knn_in_clusters(user_emb, clusters, index, seen_ids, k=40)

    # 3. Rerank and select
    results = rerank_and_select(candidates, n=n)

    # Edge case: if fewer than n, fall back to all clusters
    if len(results) < n:
        all_cluster_ids = list(range(index.centroids.shape[0]))
        # Collect IDs already selected to avoid duplicates
        selected_ids = {r["id"] for r in results}
        expanded_seen = seen_ids | selected_ids

        all_candidates = knn_in_clusters(
            user_emb, all_cluster_ids, index, expanded_seen, k=40
        )
        extra = rerank_and_select(all_candidates, n=n - len(results))
        results.extend(extra)

    return results
