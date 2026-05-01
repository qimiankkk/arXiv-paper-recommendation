from types import SimpleNamespace

import numpy as np

from recommender.config import DAILY_FEED_SIZE
from recommender.engine import recommend


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    return (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).astype(np.float32)


def _fake_index(total_clusters: int = 24, papers_per_cluster: int = 3):
    angles = np.linspace(0.0, 0.7, total_clusters, dtype=np.float32)
    centroids = _unit_rows(
        np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    )

    embeddings = []
    cluster_ids = []
    paper_meta = []
    for cluster_id, centroid in enumerate(centroids):
        for paper_i in range(papers_per_cluster):
            embeddings.append(centroid)
            cluster_ids.append(cluster_id)
            paper_id = f"c{cluster_id}-p{paper_i}"
            paper_meta.append(
                {
                    "id": paper_id,
                    "title": f"Paper {paper_id}",
                    "abstract": "A useful paper.",
                    "categories": ["cs.LG"],
                    "update_date": "2026-01-01",
                    "cluster_id": cluster_id,
                }
            )

    return SimpleNamespace(
        embeddings=np.asarray(embeddings, dtype=np.float32),
        cluster_ids=np.asarray(cluster_ids, dtype=np.int32),
        centroids=centroids,
        paper_meta=paper_meta,
    )


def test_recommend_returns_daily_feed_from_fake_index():
    recs = recommend(
        np.array([[1.0, 0.0]], dtype=np.float32),
        seen_ids=set(),
        index=_fake_index(),
        diversity=0.5,
        n=DAILY_FEED_SIZE,
    )

    assert len(recs) == DAILY_FEED_SIZE
    assert all("rec_score" in rec for rec in recs)


def test_recommend_handles_diversity_extremes_and_single_centroid():
    index = _fake_index()
    centroids = np.array([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32)
    centroids = _unit_rows(centroids)

    focused = recommend(centroids, seen_ids=set(), index=index, diversity=0.0)
    broad = recommend(centroids, seen_ids=set(), index=index, diversity=1.0)
    single = recommend(centroids[:1], seen_ids=set(), index=index, diversity=0.5)

    assert 0 < len(focused) <= DAILY_FEED_SIZE
    assert 0 < len(broad) <= DAILY_FEED_SIZE
    assert 0 < len(single) <= DAILY_FEED_SIZE
