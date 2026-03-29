"""Reranking and diversity filtering for recommendation candidates.

Takes the raw KNN candidates and applies:
1. Recency boost — newer papers get a score bonus.
2. Diversity filter — ensures selected papers come from different clusters.
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
    candidates: list[tuple[float, dict]],
    recency_weight: float = 0.25,
    n: int = 3,
) -> list[dict]:
    """Rerank candidates by combined similarity + recency, then select diverse top-n.

    Args:
        candidates: List of (similarity_score, paper_meta_dict) tuples
            from the retrieval stage.
        recency_weight: Weight of the recency bonus in the final score.
        n: Number of papers to select.

    Returns:
        List of up to n paper_meta dicts, each with an added "rec_score" key.
    """
    # Score each candidate
    scored: list[tuple[float, dict]] = []
    for sim_score, meta in candidates:
        bonus = recency_weight * recency_score(meta.get("update_date", ""))
        final_score = sim_score + bonus
        meta["rec_score"] = final_score
        scored.append((final_score, meta))

    # Sort by final score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Diversity pass: at most one paper per cluster
    selected: list[dict] = []
    used_clusters: set[int] = set()
    for _score, meta in scored:
        cid = meta.get("cluster_id")
        if cid in used_clusters:
            continue
        used_clusters.add(cid)
        selected.append(meta)
        if len(selected) >= n:
            break

    return selected
