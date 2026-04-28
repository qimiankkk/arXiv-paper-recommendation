from __future__ import annotations

from recommender.rerank import rerank_and_select


def _meta(pid: str, cluster_id: int) -> dict:
    return {
        "id": pid,
        "title": pid,
        "abstract": "",
        "cluster_id": cluster_id,
        "update_date": "2026-01-01",
    }


def test_balances_slots_across_centroids_before_backfill() -> None:
    # 5 slots, 2 centroids -> near-even allocation (2/3 split).
    candidates = [
        # centroid 1 strong candidates
        (0.99, _meta("c1_a", 1), 1),
        (0.98, _meta("c1_b", 2), 1),
        (0.97, _meta("c1_c", 3), 1),
        (0.96, _meta("c1_d", 4), 1),
        # centroid 0 candidates
        (0.80, _meta("c0_a", 10), 0),
        (0.79, _meta("c0_b", 11), 0),
        (0.78, _meta("c0_c", 12), 0),
    ]

    selected = rerank_and_select(candidates, k_u=2, diversity=0.0, n=5, recency_weight=0.0)
    ids = [m["id"] for m in selected]

    # Should keep counts near-even (difference <= 1).
    c0 = {"c0_a", "c0_b", "c0_c"}
    c1 = {"c1_a", "c1_b", "c1_c", "c1_d"}
    c0_count = sum(1 for i in ids if i in c0)
    c1_count = sum(1 for i in ids if i in c1)
    assert c0_count + c1_count == 5
    assert abs(c0_count - c1_count) <= 1


def test_backfills_when_one_centroid_cannot_meet_quota() -> None:
    # 5 slots, 2 centroids -> quotas [3, 2], but centroid 0 has only 1 candidate.
    candidates = [
        (0.99, _meta("c1_a", 1), 1),
        (0.98, _meta("c1_b", 2), 1),
        (0.97, _meta("c1_c", 3), 1),
        (0.96, _meta("c1_d", 4), 1),
        (0.95, _meta("c1_e", 5), 1),
        (0.70, _meta("c0_a", 10), 0),
    ]

    selected = rerank_and_select(candidates, k_u=2, diversity=0.0, n=5, recency_weight=0.0)
    ids = [m["id"] for m in selected]
    assert len(ids) == 5
    assert "c0_a" in ids


def test_diversity_zero_allows_same_cluster_multiple_times() -> None:
    # With diversity=0, cluster-id uniqueness should be disabled.
    candidates = [
        (0.99, _meta("a1", 7), 0),
        (0.98, _meta("a2", 7), 0),
        (0.97, _meta("b1", 8), 1),
        (0.96, _meta("b2", 9), 1),
    ]

    selected = rerank_and_select(candidates, k_u=2, diversity=0.0, n=4, recency_weight=0.0)
    ids = [m["id"] for m in selected]
    assert len(ids) == 4
    assert "a1" in ids and "a2" in ids


def test_like_driven_quota_guarantees_one_per_centroid_when_possible() -> None:
    # n=5, k_u=2, likes [4,1] -> should favor centroid 0 and keep >=1 for centroid 1.
    candidates = [
        (0.99, _meta("c0_a", 1), 0),
        (0.98, _meta("c0_b", 2), 0),
        (0.97, _meta("c0_c", 3), 0),
        (0.96, _meta("c0_d", 4), 0),
        (0.95, _meta("c1_a", 5), 1),
        (0.94, _meta("c1_b", 6), 1),
    ]

    selected = rerank_and_select(
        candidates,
        k_u=2,
        diversity=0.0,
        n=5,
        recency_weight=0.0,
        centroid_like_counts=[4, 1],
    )
    nearest = [m["nearest_centroid_idx"] for m in selected]
    assert nearest.count(0) > nearest.count(1)
    assert nearest.count(1) >= 1


def test_like_driven_order_puts_highest_like_centroid_first() -> None:
    candidates = [
        (0.99, _meta("c1_top", 1), 1),
        (0.98, _meta("c0_top", 2), 0),
        (0.97, _meta("c1_second", 3), 1),
        (0.96, _meta("c0_second", 4), 0),
    ]

    selected = rerank_and_select(
        candidates,
        k_u=2,
        diversity=0.0,
        n=4,
        recency_weight=0.0,
        centroid_like_counts=[1, 10],
    )
    nearest = [m["nearest_centroid_idx"] for m in selected]
    assert nearest[:2] == [1, 1]
