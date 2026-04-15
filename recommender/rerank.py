"""Reranking and diversity filtering for recommendation candidates.

Takes the raw KNN candidates and applies:
1. Recency boost — newer papers get a score bonus.
2. Diversity filter — ensures selected papers come from different clusters.
3. δ-aware centroid coverage — when δ > 0.5 and k_u > 1, holds output slots
   open for uncovered user centroids before filling by score alone.
"""

from __future__ import annotations

from datetime import datetime
from math import exp


def recency_score(published_date: str, halflife_days: float = 30.0) -> float:
    """Compute a recency bonus score for a paper based on its publication date.

    More recent papers get higher scores, decaying exponentially.

    Args:
        published_date: ISO format date string from paper_meta["update_date"].
        halflife_days: Controls how fast the recency score decays.
            Default 30.0 days.

    Returns:
        Float in (0, 1]. Recent papers -> ~1.0, old papers -> small positive.
    """
    try:
        published = datetime.fromisoformat(published_date)
    except (ValueError, TypeError):
        # If date can't be parsed, return a neutral mid-range score
        return 0.5

    age_days = (datetime.now() - published).days
    # Clamp age to max 365 days to avoid near-zero scores on old papers
    age_days = min(age_days, 365)
    age_days = max(age_days, 0)

    return exp(-age_days / halflife_days)


def rerank_and_select(
    candidates: list[tuple[float, dict, int]],
    k_u: int = 1,
    diversity: float = 0.5,
    recency_weight: float = 0.25,
    n: int = 5,
) -> list[dict]:
    """Rerank candidates and select diverse top-n papers.

    Scoring: final_score = similarity + recency_weight * recency(date).
    Diversity: at most one paper per k-means cluster (always enforced).
    When δ > 0.5 and k_u > 1: holds slots for uncovered user centroids
    before filling remaining spots by score.

    Args:
        candidates: List of (sim_score, paper_meta, nearest_centroid_idx).
        k_u: Number of user centroids.
        diversity: The δ slider value, 0.0–1.0.
        recency_weight: Weight of recency bonus.
        n: Papers to select. Default 5.

    Returns:
        List of up to n paper_meta dicts, each with "rec_score" added.
    """
    scored: list[tuple[float, dict, int]] = []
    for sim_score, meta, nearest_ci in candidates:
        bonus = recency_weight * recency_score(meta.get("update_date", ""))
        final = sim_score + bonus
        meta["rec_score"] = final
        scored.append((final, meta, nearest_ci))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected: list[dict] = []
    used_clusters: set[int] = set()
    covered_centroids: set[int] = set()

    for _score, meta, nearest_ci in scored:
        cid = meta.get("cluster_id")
        if cid in used_clusters:
            continue

        # Lever 2: when δ > 0.5 and we haven't filled k_u slots yet,
        # skip papers that re-cover an already-covered centroid.
        if (diversity > 0.5
                and k_u > 1
                and nearest_ci in covered_centroids
                and len(covered_centroids) < k_u):
            continue

        selected.append(meta)
        used_clusters.add(cid)
        covered_centroids.add(nearest_ci)
        if len(selected) >= n:
            break

    return selected
