"""User profile initialization and EMA-based embedding updates.

Handles two key operations:
1. Cold-start: construct an initial user embedding from selected topic categories.
2. Feedback update: shift the user embedding toward/away from a paper via EMA.

All output embeddings are guaranteed unit-norm.
"""

from __future__ import annotations

import numpy as np

FEEDBACK_WEIGHTS: dict[str, float] = {
    "like": 1.0,
    "save": 1.5,
    "skip": -0.3,
}

EMA_ALPHA: float = 0.15


def init_embedding_from_topics(
    selected_categories: list[str],
    category_centroids: dict[str, np.ndarray],
) -> np.ndarray:
    """Create an initial user embedding from selected arXiv category topics.

    Args:
        selected_categories: List of arXiv category strings the user picked
            during onboarding, e.g. ["cs.LG", "cs.CL"].
        category_centroids: Dict mapping category string to its unit-norm
            centroid vector (shape (768,), float32).

    Returns:
        Unit-norm embedding of shape (768,), float32.
    """
    vectors = [
        category_centroids[cat]
        for cat in selected_categories
        if cat in category_centroids
    ]

    if not vectors:
        # Fallback: no selected category found in corpus
        print("WARNING: No selected categories found in corpus. Using fallback centroid.")
        first_centroid = next(iter(category_centroids.values()))
        return first_centroid.astype(np.float32).copy()

    mean_vec = np.mean(vectors, axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm < 1e-8:
        first_centroid = next(iter(category_centroids.values()))
        return first_centroid.astype(np.float32).copy()

    return (mean_vec / norm).astype(np.float32)


def apply_feedback(
    user_embedding: np.ndarray,
    paper_embedding: np.ndarray,
    signal: str,
    alpha: float = EMA_ALPHA,
) -> np.ndarray:
    """Update the user embedding via exponential moving average after feedback.

    Args:
        user_embedding: Current user embedding, shape (768,), float32, unit-norm.
        paper_embedding: Embedding of the paper the user interacted with,
            shape (768,), float32, unit-norm.
        signal: One of "like", "save", "skip".
        alpha: EMA smoothing factor. Default 0.15.

    Returns:
        Updated user embedding, shape (768,), float32, unit-norm.
        If the resulting norm is < 1e-8 (degenerate case), returns
        the original user_embedding unchanged as a safety guard.
    """
    w = FEEDBACK_WEIGHTS[signal]
    raw = (1 - alpha) * user_embedding + alpha * w * paper_embedding
    norm = np.linalg.norm(raw)
    if norm < 1e-8:
        return user_embedding
    return (raw / norm).astype(np.float32)
