from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np

import recommender.engine as engine
from recommender.config import (
    DAILY_CANDIDATE_POOL_SIZE,
    DAILY_FEED_SIZE,
    DAILY_MAX_PER_CLUSTER,
)
from recommender.retrieve import find_nearest_clusters
from user.db import (
    create_user,
    get_feedback_counts,
    get_interacted_paper_count,
    get_seen_ids,
    init_db,
    log_feedback,
    mark_papers_seen,
)


def _candidate(
    paper_id: str,
    cluster_id: int,
    score: float,
    nearest_centroid: int = 0,
) -> tuple[float, dict, int]:
    return (
        score,
        {
            "id": paper_id,
            "title": paper_id,
            "abstract": "",
            "update_date": "2026-01-01",
            "cluster_id": cluster_id,
        },
        nearest_centroid,
    )


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def _index(total_clusters: int = 40) -> SimpleNamespace:
    centroids = np.stack(
        [
            np.array([1.0 - i * 0.01, 0.0], dtype=np.float32)
            for i in range(total_clusters)
        ]
    )
    return SimpleNamespace(centroids=centroids)


def test_daily_feed_constants_are_sized_for_twenty_papers():
    assert DAILY_FEED_SIZE == 20
    assert DAILY_CANDIDATE_POOL_SIZE == 200
    assert DAILY_CANDIDATE_POOL_SIZE > DAILY_FEED_SIZE
    assert DAILY_MAX_PER_CLUSTER == 2


def test_candidate_pool_size_is_used_before_reranking(monkeypatch):
    captured_k: list[int] = []
    candidates = [_candidate(f"p{i}", i // 2, 1.0 - i * 0.001) for i in range(40)]

    monkeypatch.setattr(
        engine,
        "find_nearest_clusters",
        lambda user_centroids, index_centroids, diversity: list(range(20)),
    )

    def fake_knn(user_centroids, target_cluster_ids, index, seen_ids, k):
        captured_k.append(k)
        return candidates[:k]

    monkeypatch.setattr(engine, "knn_in_clusters", fake_knn)

    recs = engine.recommend(
        np.array([[1.0, 0.0]], dtype=np.float32),
        seen_ids=set(),
        index=_index(),
    )

    assert captured_k == [DAILY_CANDIDATE_POOL_SIZE]
    assert len(recs) == DAILY_FEED_SIZE


def test_cluster_budget_follows_daily_formula_for_single_centroid():
    user_centroids = np.array([[1.0, 0.0]], dtype=np.float32)
    index_centroids = np.stack(
        [np.array([1.0 - i * 0.01, 0.0], dtype=np.float32) for i in range(20)]
    )

    assert len(find_nearest_clusters(user_centroids, index_centroids, 0.0)) == 4
    assert len(find_nearest_clusters(user_centroids, index_centroids, 0.5)) == 8
    assert len(find_nearest_clusters(user_centroids, index_centroids, 1.0)) == 12


def test_cluster_budget_clamps_out_of_range_diversity_values():
    user_centroids = np.array([[1.0, 0.0]], dtype=np.float32)
    index_centroids = np.stack(
        [np.array([1.0 - i * 0.01, 0.0], dtype=np.float32) for i in range(20)]
    )

    assert len(find_nearest_clusters(user_centroids, index_centroids, -1.0)) == 4
    assert len(find_nearest_clusters(user_centroids, index_centroids, 2.0)) == 12


def test_final_feed_never_has_more_than_two_papers_per_kmeans_cluster():
    candidates = [
        _candidate(f"p{i}", cluster_id=i // 3, score=1.0 - i * 0.001)
        for i in range(60)
    ]

    recs, _early, _selected = engine.select_with_relaxation(
        candidates,
        k_u=1,
        diversity=0.5,
        n=DAILY_FEED_SIZE,
        seen_ids=set(),
    )

    counts = Counter(rec["cluster_id"] for rec in recs)
    assert len(recs) == DAILY_FEED_SIZE
    assert max(counts.values()) <= DAILY_MAX_PER_CLUSTER


def test_delta_zero_keeps_base_score_order_when_clusters_allow_it():
    candidates = [
        _candidate("c0-best", 0, 1.0, nearest_centroid=0),
        _candidate("c1-second", 1, 0.95, nearest_centroid=1),
        _candidate("c2-third", 2, 0.90, nearest_centroid=2),
    ]

    recs, _covered, _selected = engine.select_with_relaxation(
        candidates,
        k_u=3,
        diversity=0.0,
        n=3,
        seen_ids=set(),
    )

    assert [rec["id"] for rec in recs] == ["c0-best", "c1-second", "c2-third"]


def test_high_diversity_promotes_undercovered_centroids_when_scores_are_close():
    candidates = [
        _candidate("c0-best", 0, 1.0, nearest_centroid=0),
        _candidate("c0-second", 1, 0.98, nearest_centroid=0),
        _candidate("c1-best", 2, 0.95, nearest_centroid=1),
        _candidate("c2-best", 3, 0.94, nearest_centroid=2),
    ]

    recs, _covered, _selected = engine.select_with_relaxation(
        candidates,
        k_u=3,
        diversity=1.0,
        n=3,
        seen_ids=set(),
    )

    assert [rec["id"] for rec in recs] == ["c0-best", "c1-best", "c2-best"]
    assert {rec["nearest_centroid_id"] for rec in recs} == {0, 1, 2}


def test_seen_papers_are_excluded_and_served_papers_are_marked_seen(tmp_path):
    db_path = str(tmp_path / "seen.db")
    init_db(db_path)
    centroids = np.stack(
        [
            _unit(np.arange(1, 769, dtype=np.float32)),
        ]
    )
    uid = create_user("Daily User", centroids, k_u=1)

    mark_papers_seen(uid, ["served-final", "served-final"])
    log_feedback(uid, "liked-paper", "like", cluster_id=1, score=0.9)
    log_feedback(uid, "liked-paper", "like", cluster_id=1, score=0.9)

    assert get_seen_ids(uid) == {"served-final", "liked-paper"}
    assert get_feedback_counts(uid) == {"like": 2, "save": 0, "skip": 0}
    assert get_interacted_paper_count(uid) == 1

    candidates = [
        _candidate("served-final", 0, 1.0),
        _candidate("fresh-paper", 1, 0.9),
    ]
    recs, _early, _selected = engine.select_with_relaxation(
        candidates,
        k_u=1,
        diversity=0.5,
        n=DAILY_FEED_SIZE,
        seen_ids=get_seen_ids(uid),
    )

    assert [rec["id"] for rec in recs] == ["fresh-paper"]


def test_debug_metadata_is_added_to_served_papers():
    recs, _early, _selected = engine.select_with_relaxation(
        [_candidate("p0", 3, 0.75, nearest_centroid=2)],
        k_u=3,
        diversity=0.7,
        n=DAILY_FEED_SIZE,
        seen_ids=set(),
    )

    rec = recs[0]
    assert rec["final_score"] == rec["rec_score"]
    assert rec["raw_similarity"] == 0.75
    assert "base_score" in rec
    assert "diversity_adjusted_score" in rec
    assert "centroid_coverage_bonus" in rec
    assert "cluster_saturation_penalty" in rec
    assert "recency_score" in rec
    assert rec["cluster_id"] == 3
    assert rec["nearest_centroid_id"] == 2
    assert "age_days" in rec
