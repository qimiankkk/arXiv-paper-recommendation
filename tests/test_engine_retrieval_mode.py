from __future__ import annotations

import numpy as np

from recommender.engine import recommend


class _DummyIndex:
    def __init__(self) -> None:
        self.centroids = np.array(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        self.cluster_ids = np.array([0, 1, 1], dtype=np.int32)
        self.embeddings = np.array(
            [
                [1.0, 0.0],   # p0: best match for user, cluster 0
                [0.7, 0.7],   # p1: weaker match, cluster 1
                [0.2, 0.98],  # p2: weak match, cluster 1
            ],
            dtype=np.float32,
        )
        self.paper_meta = [
            {
                "id": "p0",
                "title": "p0",
                "abstract": "",
                "cluster_id": 0,
                "update_date": "2026-01-01",
            },
            {
                "id": "p1",
                "title": "p1",
                "abstract": "",
                "cluster_id": 1,
                "update_date": "2026-01-01",
            },
            {
                "id": "p2",
                "title": "p2",
                "abstract": "",
                "cluster_id": 1,
                "update_date": "2026-01-01",
            },
        ]


def test_diversity_zero_uses_global_similarity_search() -> None:
    idx = _DummyIndex()
    user = np.array([[1.0, 0.0]], dtype=np.float32)

    recs = recommend(
        user_centroids=user,
        seen_ids=set(),
        index=idx,
        diversity=0.0,
        n=1,
    )

    assert recs
    assert recs[0]["id"] == "p0"
