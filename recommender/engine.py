"""Top-level recommendation engine.

Provides a single recommend() entry point that orchestrates retrieval and
reranking to produce the final list of recommended papers.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable

import numpy as np

from pipeline.index import PaperIndex
from recommender.config import (
    DAILY_CANDIDATE_POOL_SIZE,
    DAILY_CENTROID_COVERAGE_WEIGHT,
    DAILY_CLUSTER_SATURATION_WEIGHT,
    DAILY_FEED_SIZE,
    DAILY_MAX_PER_CLUSTER,
)
from recommender.diversity import clamp_diversity
from recommender.retrieve import find_nearest_clusters, knn_in_clusters
from recommender.rerank import paper_age_days, recency_score


logger = logging.getLogger(__name__)

TARGET_RECOMMENDATIONS = DAILY_FEED_SIZE
INITIAL_TOP_M = DAILY_CANDIDATE_POOL_SIZE
EXPANDED_TOP_M = DAILY_CANDIDATE_POOL_SIZE
MAX_INITIAL_CLUSTERS = 12
MAX_EXPANDED_CLUSTERS = 16
RELAXED_CLUSTER_CAP = DAILY_MAX_PER_CLUSTER


def _normalize_cluster_ids(cluster_ids: Iterable[int] | np.ndarray) -> set[int]:
    """Normalize cluster IDs from numpy/list/set forms into plain ints."""
    return {int(cid) for cid in list(cluster_ids)}


def _rank_selected_clusters(
    index: PaperIndex,
    user_centroids: np.ndarray,
    cluster_ids: set[int],
) -> list[int]:
    """Rank a cluster subset by its best similarity to any user centroid."""
    if not cluster_ids:
        return []
    sims = index.centroids @ user_centroids.T
    max_sims = sims.max(axis=1)
    return sorted(cluster_ids, key=lambda cid: float(max_sims[cid]), reverse=True)


def _cap_selected_clusters(
    index: PaperIndex,
    user_centroids: np.ndarray,
    cluster_ids: Iterable[int] | np.ndarray,
    max_total_clusters: int,
) -> list[int]:
    """Keep an existing cluster selection under a hard cluster cap."""
    selected = _normalize_cluster_ids(cluster_ids)
    if len(selected) <= max_total_clusters:
        return sorted(selected)
    return sorted(
        _rank_selected_clusters(index, user_centroids, selected)[:max_total_clusters]
    )


def expand_clusters_near_user(
    index: PaperIndex,
    user_centroids: np.ndarray,
    selected_clusters: Iterable[int] | np.ndarray,
    max_total_clusters: int = MAX_EXPANDED_CLUSTERS,
) -> list[int]:
    """
    Expand selected retrieval clusters by adding the nearest k-means clusters
    to the user centroids, but never exceed max_total_clusters.
    """
    total_clusters = int(index.centroids.shape[0])
    if total_clusters <= 0:
        return []

    # Leave at least one cluster unscanned when possible, making the no-all-
    # clusters fallback invariant explicit even for small synthetic indexes.
    effective_cap = min(max_total_clusters, max(1, total_clusters - 1))
    expanded = _normalize_cluster_ids(selected_clusters)

    if len(expanded) > effective_cap:
        expanded = set(
            _rank_selected_clusters(index, user_centroids, expanded)[:effective_cap]
        )

    rankings: list[list[int]] = []
    for centroid in user_centroids:
        sims = index.centroids @ centroid
        rankings.append(np.argsort(sims)[::-1].astype(int).tolist())

    depth = 0
    while len(expanded) < effective_cap:
        added = False
        for ranking in rankings:
            if depth >= len(ranking):
                continue
            cid = int(ranking[depth])
            if cid not in expanded:
                expanded.add(cid)
                added = True
                if len(expanded) >= effective_cap:
                    break
        if not added and all(depth >= len(ranking) for ranking in rankings):
            break
        depth += 1

    assert len(expanded) <= max_total_clusters
    if total_clusters > 1:
        assert len(expanded) < total_clusters
    return sorted(expanded)


def _score_candidates(
    candidates: list[tuple[float, dict, int]],
    recency_weight: float = 0.25,
) -> list[tuple[float, dict, int]]:
    scored: list[tuple[float, dict, int]] = []
    for sim_score, meta, nearest_ci in candidates:
        recency = recency_score(meta.get("update_date", ""))
        base_score = sim_score + recency_weight * recency
        enriched_meta = dict(meta)
        enriched_meta["rec_score"] = base_score
        enriched_meta["final_score"] = base_score
        enriched_meta["base_score"] = base_score
        enriched_meta["raw_similarity"] = float(sim_score)
        enriched_meta["recency_score"] = float(recency)
        enriched_meta["nearest_centroid_id"] = int(nearest_ci)
        enriched_meta["age_days"] = paper_age_days(meta.get("update_date", ""))
        scored.append((base_score, enriched_meta, nearest_ci))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _candidate_is_available(
    meta: dict,
    seen_ids: set[str],
    selected_ids: set[str],
    cluster_counts: Counter,
    max_per_cluster: int,
) -> bool:
    pid = meta.get("id")
    if pid in seen_ids or pid in selected_ids:
        return False
    cid = meta.get("cluster_id")
    if cluster_counts[cid] >= max_per_cluster:
        return False
    return True


def _append_candidate(
    selected: list[dict],
    selected_ids: set[str],
    cluster_counts: Counter,
    covered_centroids: set[int],
    thread_counts: Counter,
    meta: dict,
    nearest_ci: int,
) -> None:
    selected.append(meta)
    pid = meta.get("id")
    if pid is not None:
        selected_ids.add(pid)
    cluster_counts[meta.get("cluster_id")] += 1
    covered_centroids.add(nearest_ci)
    thread_counts[nearest_ci] += 1


def _diversity_adjusted_score(
    base_score: float,
    meta: dict,
    nearest_ci: int,
    diversity: float,
    thread_counts: Counter,
    cluster_counts: Counter,
    max_per_cluster: int,
) -> tuple[float, float, float]:
    """Score a candidate using centroid coverage and cluster saturation."""
    coverage_bonus = 1.0 / (1.0 + thread_counts[nearest_ci])
    cluster_id = meta.get("cluster_id")
    saturation = cluster_counts[cluster_id] / max_per_cluster
    adjusted_score = (
        base_score
        + diversity * DAILY_CENTROID_COVERAGE_WEIGHT * coverage_bonus
        - diversity * DAILY_CLUSTER_SATURATION_WEIGHT * saturation
    )
    return adjusted_score, coverage_bonus, saturation


def select_with_relaxation(
    candidates: list[tuple[float, dict, int]],
    k_u: int,
    diversity: float,
    n: int = TARGET_RECOMMENDATIONS,
    seen_ids: set[str] | None = None,
) -> tuple[list[dict], int, int]:
    """Select recommendations with delta-weighted centroid coverage."""
    seen_ids = seen_ids or set()
    diversity = clamp_diversity(diversity)
    scored = _score_candidates(candidates)

    selected: list[dict] = []
    selected_ids: set[str] = set()
    cluster_counts: Counter = Counter()
    thread_counts: Counter = Counter()
    covered_centroids: set[int] = set()

    while len(selected) < n:
        best: tuple[float, float, float, dict, int] | None = None
        for base_score, meta, nearest_ci in scored:
            if not _candidate_is_available(
                meta,
                seen_ids,
                selected_ids,
                cluster_counts,
                DAILY_MAX_PER_CLUSTER,
            ):
                continue
            adjusted_score, coverage_bonus, saturation = _diversity_adjusted_score(
                base_score=base_score,
                meta=meta,
                nearest_ci=nearest_ci,
                diversity=diversity,
                thread_counts=thread_counts,
                cluster_counts=cluster_counts,
                max_per_cluster=DAILY_MAX_PER_CLUSTER,
            )
            if best is None or adjusted_score > best[0]:
                best = (
                    adjusted_score,
                    coverage_bonus,
                    saturation,
                    meta,
                    nearest_ci,
                )

        if best is None:
            break

        adjusted_score, coverage_bonus, saturation, meta, nearest_ci = best
        selected_meta = dict(meta)
        selected_meta["final_score"] = float(adjusted_score)
        selected_meta["rec_score"] = float(adjusted_score)
        selected_meta["diversity_adjusted_score"] = float(adjusted_score)
        selected_meta["centroid_coverage_bonus"] = float(coverage_bonus)
        selected_meta["cluster_saturation_penalty"] = float(saturation)

        _append_candidate(
            selected,
            selected_ids,
            cluster_counts,
            covered_centroids,
            thread_counts,
            selected_meta,
            nearest_ci,
        )

    return selected[:n], len(covered_centroids), len(selected)


def _dedupe_candidates(
    candidates: list[tuple[float, dict, int]],
    seen_ids: set[str],
) -> list[tuple[float, dict, int]]:
    """Keep one candidate per paper ID, preferring the highest raw similarity."""
    by_id: dict[str, tuple[float, dict, int]] = {}
    anonymous: list[tuple[float, dict, int]] = []

    for candidate in candidates:
        sim_score, meta, _nearest_ci = candidate
        pid = meta.get("id")
        if pid in seen_ids:
            continue
        if pid is None:
            anonymous.append(candidate)
            continue
        if pid not in by_id or sim_score > by_id[pid][0]:
            by_id[pid] = candidate

    deduped = list(by_id.values()) + anonymous
    deduped.sort(key=lambda item: item[0], reverse=True)
    return deduped


def recommend(
    user_centroids: np.ndarray,
    seen_ids: set[str],
    index: PaperIndex,
    diversity: float = 0.5,
    n: int = TARGET_RECOMMENDATIONS,
) -> list[dict]:
    """Generate n paper recommendations for a user.

    Args:
        user_centroids: Shape (k_u, 768), float32, unit-norm rows.
        seen_ids: Set of arXiv paper IDs already seen.
        index: Loaded PaperIndex.
        diversity: δ slider value, 0.0–1.0.
        n: Papers to recommend. Default 20.

    Returns:
        List of up to n paper_meta dicts with "rec_score" added.
    """
    diversity = clamp_diversity(diversity)
    k_u = user_centroids.shape[0]

    # 1. Cluster selection (δ controls budget)
    clusters = _cap_selected_clusters(
        index,
        user_centroids,
        find_nearest_clusters(user_centroids, index.centroids, diversity),
        MAX_INITIAL_CLUSTERS,
    )
    initial_clusters = list(clusters)

    # 2. KNN within those clusters
    candidates = knn_in_clusters(
        user_centroids,
        clusters,
        index,
        seen_ids,
        k=INITIAL_TOP_M,
    )

    # 3. Rerank and apply daily-feed diversity constraints
    results, strict_count, relaxed_count = select_with_relaxation(
        candidates,
        k_u=k_u,
        diversity=diversity,
        n=n,
        seen_ids=seen_ids,
    )

    bounded_expansion_used = False
    final_clusters = list(initial_clusters)
    if len(results) < n:
        expanded_clusters = expand_clusters_near_user(
            index=index,
            user_centroids=user_centroids,
            selected_clusters=initial_clusters,
            max_total_clusters=MAX_EXPANDED_CLUSTERS,
        )
        additional_clusters = sorted(set(expanded_clusters) - set(initial_clusters))
        if additional_clusters:
            bounded_expansion_used = True
            final_clusters = list(expanded_clusters)
            extra_candidates = knn_in_clusters(
                user_centroids,
                additional_clusters,
                index,
                seen_ids,
                k=EXPANDED_TOP_M,
            )
            candidates = _dedupe_candidates(candidates + extra_candidates, seen_ids)
            results, strict_count, relaxed_count = select_with_relaxation(
                candidates,
                k_u=k_u,
                diversity=diversity,
                n=n,
                seen_ids=seen_ids,
            )

    logger.info(
        "recommendations: initial_clusters=%d expanded_clusters=%d candidates=%d "
        "early_coverage_selected=%d capped_selected=%d final_selected=%d "
        "bounded_expansion_used=%s",
        len(initial_clusters),
        len(final_clusters),
        len(candidates),
        strict_count,
        relaxed_count,
        len(results),
        bounded_expansion_used,
    )

    return results[:n]
