"""User profile initialization and EMA-based centroid updates.

Handles two key operations:
1. Cold-start: construct a multi-vector user profile from selected topics
   and optional Scholar paper embeddings.
2. Feedback update: shift the nearest user centroid toward/away from a paper via EMA.

All output centroids are guaranteed unit-norm rows.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

FEEDBACK_WEIGHTS: dict[str, float] = {
    "like": 1.0,
    "save": 1.5,
    "skip": -0.3,
}

EMA_ALPHA: float = 0.15


def init_user_profile(
    topic_keys: list[str],
    category_centroids: dict[str, np.ndarray],
    paper_embeddings: np.ndarray | None = None,
    max_k: int = 3,
) -> np.ndarray:
    """Initialize a multi-vector user profile from topics and optional papers.

    Args:
        topic_keys: Selected arXiv category strings, e.g. ["cs.LG", "cs.CL"].
        category_centroids: Dict mapping category string to unit-norm centroid (768,).
        paper_embeddings: Optional (n_papers, 768) float32 unit-norm embeddings
            from a Scholar or GitHub profile upload. None if tags only.
        max_k: Maximum number of user centroids. Default 3.

    Returns:
        Unit-norm centroids of shape (k_u, 768), where k_u = min(max_k, len(topic_keys)).
    """
    # Step 1: collect seed vectors
    tag_vecs = [category_centroids[t] for t in topic_keys if t in category_centroids]
    if not tag_vecs:
        fallback = next(iter(category_centroids.values()))
        return fallback.astype(np.float32).copy().reshape(1, 768)

    seeds = np.stack(tag_vecs)  # (n_tags, 768)
    if paper_embeddings is not None and len(paper_embeddings) > 0:
        seeds = np.vstack([paper_embeddings, seeds])

    # Step 2: cluster into k_u centroids
    k_u = min(max_k, len(topic_keys))
    if k_u <= 1 or len(seeds) <= 1:
        mean_vec = seeds.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm < 1e-8:
            return seeds[:1].copy()
        return (mean_vec / norm).astype(np.float32).reshape(1, 768)

    # If fewer seeds than k_u, reduce k_u to avoid KMeans error
    k_u = min(k_u, len(seeds))
    km = KMeans(n_clusters=k_u, n_init=10, random_state=42)
    km.fit(seeds)
    centroids = km.cluster_centers_.astype(np.float32)

    # Normalize each row to unit length
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    centroids = centroids / norms

    return centroids


def apply_feedback(
    centroids: np.ndarray,
    paper_embedding: np.ndarray,
    signal: str,
    alpha: float = EMA_ALPHA,
) -> np.ndarray:
    """Update the nearest user centroid via EMA after a feedback event.

    Only the centroid closest to the paper is modified. All other
    centroids remain unchanged, preserving distinct research threads.

    Args:
        centroids: Shape (k_u, 768), float32, unit-norm rows.
        paper_embedding: Shape (768,), float32, unit-norm.
        signal: One of "like", "save", "skip".
        alpha: EMA smoothing factor. Default 0.15.

    Returns:
        Updated centroids, same shape. The modified row is re-normalized.
        If the update would produce a degenerate vector (norm < 1e-8),
        the original centroids are returned unchanged.
    """
    w = FEEDBACK_WEIGHTS[signal]
    i_star = int(np.argmax(centroids @ paper_embedding))

    raw = (1 - alpha) * centroids[i_star] + alpha * w * paper_embedding
    norm = np.linalg.norm(raw)
    if norm < 1e-8:
        return centroids

    updated = centroids.copy()
    updated[i_star] = (raw / norm).astype(np.float32)
    return updated
