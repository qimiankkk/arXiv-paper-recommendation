"""User profile initialization and EMA-based centroid updates.

Handles two key operations:
1. Cold-start: construct user centroids from one onboarding mode at a time
   (natural-language description OR selected topic tags OR Scholar top papers).
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


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalized float32 matrix."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (mat / norms).astype(np.float32)


def init_user_profile_from_topics(
    topic_keys: list[str],
    category_centroids: dict[str, np.ndarray],
    max_k: int = 3,
) -> np.ndarray:
    """Initialize a multi-vector user profile from selected topic tags only.

    Args:
        topic_keys: Selected arXiv category strings, e.g. ["cs.LG", "cs.CL"].
        category_centroids: Dict mapping category string to unit-norm centroid (768,).
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

    return _normalize_rows(centroids)


def init_user_profile_from_description(description_embedding: np.ndarray) -> np.ndarray:
    """Initialize user profile from a single natural-language embedding.

    Args:
        description_embedding: Shape (768,) embedding vector from text prompt.

    Returns:
        Unit-norm centroids of shape (1, 768).
    """
    vec = description_embedding.astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        fallback = np.zeros_like(vec, dtype=np.float32)
        fallback[0] = 1.0
        return fallback.reshape(1, -1)
    return (vec / norm).reshape(1, -1).astype(np.float32)


def init_user_profile_from_scholar_top3(
    scholar_embeddings: np.ndarray,
    category_centroids: dict[str, np.ndarray],
) -> np.ndarray:
    """Initialize user profile from top-3 Scholar papers (no topic mixing).

    Each Scholar paper is assigned to the nearest category centroid by cosine
    similarity. Papers in the same nearest category are averaged into one
    centroid; different categories become separate centroids.

    Args:
        scholar_embeddings: Shape (n, 768), usually n=3.
        category_centroids: Dict mapping category string to unit-norm centroid.

    Returns:
        Unit-norm centroids of shape (k_u, 768), where k_u is the number of
        distinct nearest categories among the input Scholar papers.
    """
    if scholar_embeddings is None or len(scholar_embeddings) == 0:
        fallback = next(iter(category_centroids.values()))
        return fallback.astype(np.float32).copy().reshape(1, -1)

    cat_keys = list(category_centroids.keys())
    cat_mat = np.stack([category_centroids[k] for k in cat_keys]).astype(np.float32)
    cat_mat = _normalize_rows(cat_mat)
    paper_mat = _normalize_rows(scholar_embeddings.astype(np.float32))

    sims = paper_mat @ cat_mat.T
    nearest_idx = np.argmax(sims, axis=1)

    grouped: dict[int, list[np.ndarray]] = {}
    for i, cat_i in enumerate(nearest_idx):
        grouped.setdefault(int(cat_i), []).append(paper_mat[i])

    centroids: list[np.ndarray] = []
    for _, vecs in grouped.items():
        mean_vec = np.mean(np.stack(vecs, axis=0), axis=0)
        centroids.append(mean_vec.astype(np.float32))

    return _normalize_rows(np.stack(centroids, axis=0))


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
