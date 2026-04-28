"""Reranking and diversity filtering for recommendation candidates.

Takes the raw KNN candidates and applies:
1. Recency boost — newer papers get a score bonus.
2. Diversity filter — ensures selected papers come from different clusters.
3. Centroid-balanced allocation — first reserves near-even slots per user
   centroid thread, then fills remaining slots by score.
"""

from __future__ import annotations

from datetime import datetime
from math import exp


def _base_even_quotas(k_u: int, n: int) -> list[int]:
    """Return near-even quotas, preserving previous behavior."""
    if k_u <= 0:
        return []
    base = n // k_u
    remainder = n % k_u
    return [base + (1 if i < remainder else 0) for i in range(k_u)]


def _quotas_from_likes(k_u: int, n: int, centroid_like_counts: list[int]) -> list[int]:
    """Build per-centroid quotas from likes with hard 1-item minimum when possible."""
    if k_u <= 0:
        return []
    if n <= 0:
        return [0 for _ in range(k_u)]
    if n < k_u:
        # Cannot satisfy "at least one each"; allocate to top-liked centroids first.
        ordered = sorted(range(k_u), key=lambda i: centroid_like_counts[i], reverse=True)
        quotas = [0 for _ in range(k_u)]
        for i in ordered[:n]:
            quotas[i] = 1
        return quotas

    quotas = [1 for _ in range(k_u)]
    remaining = n - k_u
    weights = [max(0, int(x)) for x in centroid_like_counts]
    total_weight = sum(weights)
    if remaining == 0 or total_weight <= 0:
        return quotas

    raw = [(remaining * w) / total_weight for w in weights]
    extra = [int(x) for x in raw]
    quotas = [q + e for q, e in zip(quotas, extra)]
    assigned = sum(extra)
    leftovers = remaining - assigned
    if leftovers > 0:
        frac_order = sorted(
            range(k_u),
            key=lambda i: (raw[i] - extra[i], weights[i], -i),
            reverse=True,
        )
        for i in frac_order[:leftovers]:
            quotas[i] += 1
    return quotas


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
    centroid_like_counts: list[int] | None = None,
) -> list[dict]:
    """Rerank candidates and select diverse top-n papers.

    Scoring: final_score = similarity + recency_weight * recency(date).
    Diversity: at most one paper per k-means cluster (always enforced).
    Allocation: slots are split as evenly as possible across user centroids
    first, then any unfilled slots are backfilled by global score.

    Args:
        candidates: List of (sim_score, paper_meta, nearest_centroid_idx).
        k_u: Number of user centroids.
        diversity: The δ slider value, 0.0–1.0.
        recency_weight: Weight of recency bonus.
        n: Papers to select. Default 5.
        centroid_like_counts: Optional like-count per centroid index.

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

    selected: list[tuple[dict, int]] = []
    used_clusters: set[int] = set()
    enforce_cluster_uniqueness = diversity > 0.0
    if k_u <= 0:
        return []

    has_like_signal = (
        centroid_like_counts is not None
        and len(centroid_like_counts) == k_u
        and any(c > 0 for c in centroid_like_counts)
    )
    if has_like_signal:
        quotas = _quotas_from_likes(k_u, n, centroid_like_counts)
    else:
        quotas = _base_even_quotas(k_u, n)
    picked_per_centroid = [0 for _ in range(k_u)]

    # Pass 1: fill per-centroid quotas by score.
    for _score, meta, nearest_ci in scored:
        if len(selected) >= n:
            break
        if nearest_ci < 0 or nearest_ci >= k_u:
            continue
        if picked_per_centroid[nearest_ci] >= quotas[nearest_ci]:
            continue
        cid = meta.get("cluster_id")
        if enforce_cluster_uniqueness and cid in used_clusters:
            continue

        selected.append((meta, nearest_ci))
        if enforce_cluster_uniqueness:
            used_clusters.add(cid)
        picked_per_centroid[nearest_ci] += 1

    # Pass 2: backfill remaining slots by global score.
    if len(selected) < n:
        selected_ids = {m["id"] for m, _ci in selected}
        for _score, meta, nearest_ci in scored:
            if len(selected) >= n:
                break
            if meta["id"] in selected_ids:
                continue
            cid = meta.get("cluster_id")
            if enforce_cluster_uniqueness and cid in used_clusters:
                continue
            selected.append((meta, nearest_ci))
            if enforce_cluster_uniqueness:
                used_clusters.add(cid)
            selected_ids.add(meta["id"])

    if has_like_signal and centroid_like_counts is not None:
        selected.sort(
            key=lambda x: (
                centroid_like_counts[x[1]] if 0 <= x[1] < len(centroid_like_counts) else 0,
                x[0].get("rec_score", 0.0),
            ),
            reverse=True,
        )

    output: list[dict] = []
    for meta, nearest_ci in selected:
        meta["nearest_centroid_idx"] = int(nearest_ci)
        output.append(meta)
    return output
