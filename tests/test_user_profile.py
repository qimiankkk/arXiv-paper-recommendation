from __future__ import annotations

import numpy as np

from user.profile import (
    init_user_profile_from_description,
    init_user_profile_from_scholar_top3,
    init_user_profile_from_topics,
)


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return (vec / max(norm, 1e-8)).astype(np.float32)


def test_topic_mode_keeps_selected_direction() -> None:
    dim = 768

    # Use orthogonal basis-like vectors to make direction checks explicit.
    nlp = np.zeros(dim, dtype=np.float32)
    nlp[0] = 1.0

    cv = np.zeros(dim, dtype=np.float32)
    cv[1] = 1.0

    category_centroids = {
        "cs.CL": _unit(nlp),
        "cs.CV": _unit(cv),
    }

    centroids = init_user_profile_from_topics(
        topic_keys=["cs.CL"],
        category_centroids=category_centroids,
    )

    assert centroids.shape == (1, dim)
    nlp_sim = float(centroids[0] @ category_centroids["cs.CL"])
    cv_sim = float(centroids[0] @ category_centroids["cs.CV"])
    assert nlp_sim > cv_sim


def test_description_mode_returns_single_unit_vector() -> None:
    dim = 768
    vec = np.zeros(dim, dtype=np.float32)
    vec[7] = 3.0
    centroid = init_user_profile_from_description(vec)
    assert centroid.shape == (1, dim)
    assert np.isclose(float(np.linalg.norm(centroid[0])), 1.0, atol=1e-6)
    assert float(centroid[0][7]) > 0.99


def test_scholar_mode_groups_by_nearest_category() -> None:
    dim = 768
    nlp = np.zeros(dim, dtype=np.float32)
    nlp[0] = 1.0
    cv = np.zeros(dim, dtype=np.float32)
    cv[1] = 1.0

    category_centroids = {
        "cs.CL": _unit(nlp),
        "cs.CV": _unit(cv),
    }

    # Three papers: two near NLP, one near CV => 2 centroids after grouping.
    p1 = _unit(nlp + 0.01 * cv)
    p2 = _unit(nlp + 0.02 * cv)
    p3 = _unit(cv + 0.01 * nlp)
    scholar_embeddings = np.stack([p1, p2, p3], axis=0)

    centroids = init_user_profile_from_scholar_top3(
        scholar_embeddings,
        category_centroids,
    )

    assert centroids.shape == (2, dim)
