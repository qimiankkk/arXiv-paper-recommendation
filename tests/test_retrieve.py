from __future__ import annotations

import numpy as np

from recommender.retrieve import find_nearest_clusters, knn_in_clusters_balanced


def test_find_nearest_clusters_returns_two_when_called_with_zero_diversity() -> None:
    # User vector is closest to centroid index 1.
    user = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    centroids = np.array(
        [
            [1.0, 0.0, 0.0],   # idx 0
            [0.0, 1.0, 0.0],   # idx 1 (nearest)
            [0.0, 0.8, 0.2],   # idx 2
        ],
        dtype=np.float32,
    )

    selected = find_nearest_clusters(user, centroids, diversity=0.0)
    assert len(selected) == 2
    assert 1 in selected


class _DummyIndex:
    def __init__(self) -> None:
        self.cluster_ids = np.array([0, 0, 0, 0], dtype=np.int32)
        # Four papers: all have higher max-sim on centroid 0 except one.
        self.embeddings = np.array(
            [
                [0.99, 0.10],  # nearest 0
                [0.98, 0.20],  # nearest 0
                [0.97, 0.30],  # nearest 0
                [0.40, 0.95],  # nearest 1
            ],
            dtype=np.float32,
        )
        self.paper_meta = [
            {"id": "p0", "title": "p0", "abstract": "", "cluster_id": 0},
            {"id": "p1", "title": "p1", "abstract": "", "cluster_id": 0},
            {"id": "p2", "title": "p2", "abstract": "", "cluster_id": 0},
            {"id": "p3", "title": "p3", "abstract": "", "cluster_id": 0},
        ]


def test_balanced_retrieval_keeps_minimum_coverage_per_centroid() -> None:
    idx = _DummyIndex()
    user_centroids = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    cands = knn_in_clusters_balanced(
        user_centroids=user_centroids,
        target_cluster_ids=[0],
        index=idx,
        seen_ids=set(),
        k=4,
        centroid_like_counts=[10, 1],
    )
    nearest = [ci for _score, _meta, ci in cands]
    assert len(cands) == 4
    assert 0 in nearest
    assert 1 in nearest
