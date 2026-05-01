"""Personalized semantic query search.

This module is intentionally separate from the daily-feed recommendation
engine. Query search is driven primarily by the user's typed query, with the
profile centroids acting as a personalization prior.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import datetime

import numpy as np

from pipeline.index import PaperIndex
from recommender.retrieve import _is_withdrawn_paper
from recommender.scoring import recency_score


QUERY_CLUSTER_BUDGET = 10
MAX_USER_CLUSTER_BUDGET = QUERY_CLUSTER_BUDGET - 1
CANDIDATE_POOL_SIZE = 800
RESULT_SIZE = 20
CLUSTER_RESULT_CAP = 3
RECENCY_TIMESCALE_DAYS = 45.0

QUERY_WEIGHT = 0.6
USER_WEIGHT = 0.25
RECENCY_WEIGHT = 0.1
LEXICAL_WEIGHT = 0.05

QUERY_EXPANSION_TEMPLATE = (
    "Research papers about {query}. Focus on relevant methods, datasets, "
    "benchmarks, applications, theoretical contributions, and recent scientific progress."
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9.+#-]*", re.IGNORECASE)


def expand_query(query: str, word_threshold: int = 30) -> str:
    """Expand short raw queries into paper-like scientific retrieval text."""
    cleaned = " ".join(query.strip().split())
    if len(cleaned.split()) < word_threshold:
        return QUERY_EXPANSION_TEMPLATE.format(query=cleaned)
    return cleaned


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def _zscore(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean()) / (values.std() + eps)


def _paper_age_days(update_date: str | None) -> int | None:
    if not update_date:
        return None
    try:
        published = datetime.fromisoformat(update_date)
    except (TypeError, ValueError):
        return None
    return max((datetime.now() - published).days, 0)


def _query_terms(query: str) -> list[str]:
    """Extract simple lexical terms from the raw user query."""
    return [term.lower() for term in _TOKEN_PATTERN.findall(query)]


def lexical_score(query: str, meta: dict) -> float:
    """Score exact lexical evidence in title/abstract for short-name queries.

    This is intentionally lightweight rather than BM25. It gives a title match
    more credit than an abstract match and adds a small phrase-match boost so
    names like "LoRA" or "SwinIR" can survive dense-embedding ambiguity.
    """
    terms = _query_terms(query)
    if not terms:
        return 0.0

    title = (meta.get("title") or "").lower()
    abstract = (meta.get("abstract") or "").lower()
    raw_query = " ".join(query.strip().lower().split())

    phrase_bonus = 0.0
    if raw_query and raw_query in title:
        phrase_bonus = 1.0
    elif raw_query and raw_query in abstract:
        phrase_bonus = 0.5

    title_hits = sum(1 for term in terms if term in title)
    abstract_hits = sum(1 for term in terms if term in abstract)
    weighted_hits = (2.0 * title_hits) + abstract_hits
    coverage_score = weighted_hits / (3.0 * len(terms))

    return float(min(1.0, max(phrase_bonus, coverage_score)))


def select_query_clusters(
    query_embedding: np.ndarray,
    index_centroids: np.ndarray,
    budget: int = QUERY_CLUSTER_BUDGET,
) -> list[int]:
    """Return the nearest k-means clusters to the query embedding."""
    if budget <= 0:
        return []
    query_embedding = _normalize_vector(query_embedding)
    sims = np.asarray(index_centroids) @ query_embedding
    top = np.argsort(sims)[::-1][:budget]
    return [int(cid) for cid in top]


def select_user_clusters(
    user_centroids: np.ndarray,
    index_centroids: np.ndarray,
    diversity: float = 0.5,
    max_budget: int = MAX_USER_CLUSTER_BUDGET,
) -> list[int]:
    """Return user-profile clusters with a diversity-controlled budget."""
    if max_budget <= 0 or user_centroids.size == 0:
        return []
    budget = min(math.ceil(2 + 4 * diversity), max_budget)
    user_centroids = _normalize_rows(user_centroids)
    sims = np.asarray(index_centroids) @ user_centroids.T
    cluster_scores = sims.max(axis=1)
    top = np.argsort(cluster_scores)[::-1][:budget]
    return [int(cid) for cid in top]


def _filter_candidate_indices(
    candidate_indices: np.ndarray,
    index: PaperIndex,
    seen_ids: set[str],
    time_filter_days: int | None,
) -> list[int]:
    filtered: list[int] = []
    for original_idx in candidate_indices:
        meta = index.paper_meta[int(original_idx)]
        if meta.get("id") in seen_ids:
            continue
        if _is_withdrawn_paper(meta):
            continue
        if time_filter_days is not None:
            age_days = _paper_age_days(meta.get("update_date"))
            if age_days is None or age_days > time_filter_days:
                continue
        filtered.append(int(original_idx))
    return filtered


def search_papers(
    query: str,
    query_embedding: np.ndarray,
    user_centroids: np.ndarray,
    index: PaperIndex,
    seen_ids: set[str] | None = None,
    diversity: float = 0.5,
    n: int = RESULT_SIZE,
    query_cluster_budget: int = QUERY_CLUSTER_BUDGET,
    candidate_pool_size: int = CANDIDATE_POOL_SIZE,
    cluster_result_cap: int = CLUSTER_RESULT_CAP,
    time_filter_days: int | None = None,
) -> list[dict]:
    """Search papers using query, user, recency, and lexical relevance.

    Final score:
        0.6 * query similarity + 0.25 * user similarity
        + 0.1 * recency + 0.05 * lexical

    Components are z-score normalized over the candidate pool before the
    weighted sum is computed.
    """
    seen_ids = seen_ids or set()
    query_embedding = _normalize_vector(query_embedding)
    user_centroids = _normalize_rows(user_centroids)

    query_clusters = select_query_clusters(
        query_embedding,
        index.centroids,
        budget=query_cluster_budget,
    )
    user_clusters = select_user_clusters(
        user_centroids,
        index.centroids,
        diversity=diversity,
        max_budget=max(0, query_cluster_budget - 1),
    )
    selected_clusters = sorted(set(query_clusters) | set(user_clusters))
    if not selected_clusters:
        return []

    cluster_mask = np.isin(index.cluster_ids, selected_clusters)
    candidate_indices = np.where(cluster_mask)[0]
    if len(candidate_indices) == 0:
        return []

    filtered_indices = _filter_candidate_indices(
        candidate_indices,
        index=index,
        seen_ids=seen_ids,
        time_filter_days=time_filter_days,
    )
    if not filtered_indices:
        return []

    candidate_embeddings = np.asarray(index.embeddings[filtered_indices])
    query_sims = candidate_embeddings @ query_embedding
    preliminary_order = np.argsort(query_sims)[::-1][:candidate_pool_size]

    pool_indices = [filtered_indices[int(i)] for i in preliminary_order]
    pool_embeddings = candidate_embeddings[preliminary_order]
    pool_query_sims = query_sims[preliminary_order]

    user_sim_matrix = pool_embeddings @ user_centroids.T
    user_sims = user_sim_matrix.max(axis=1)
    nearest_threads = user_sim_matrix.argmax(axis=1)
    recency_scores = np.array(
        [
            recency_score(
                index.paper_meta[paper_idx].get("update_date", ""),
                halflife_days=RECENCY_TIMESCALE_DAYS,
            )
            for paper_idx in pool_indices
        ],
        dtype=np.float32,
    )
    lexical_scores = np.array(
        [lexical_score(query, index.paper_meta[paper_idx]) for paper_idx in pool_indices],
        dtype=np.float32,
    )

    final_scores = (
        QUERY_WEIGHT * _zscore(pool_query_sims)
        + USER_WEIGHT * _zscore(user_sims)
        + RECENCY_WEIGHT * _zscore(recency_scores)
        + LEXICAL_WEIGHT * _zscore(lexical_scores)
    )
    ranked_order = np.argsort(final_scores)[::-1]

    selected: list[dict] = []
    selected_ids: set[str] = set()
    per_cluster_counts: dict[int, int] = defaultdict(int)

    for rank_idx in ranked_order:
        paper_idx = pool_indices[int(rank_idx)]
        meta = dict(index.paper_meta[paper_idx])
        paper_id = meta.get("id")
        cluster_id = int(meta.get("cluster_id", index.cluster_ids[paper_idx]))
        if paper_id in selected_ids:
            continue
        if per_cluster_counts[cluster_id] >= cluster_result_cap:
            continue

        score = float(final_scores[rank_idx])
        meta["cluster_id"] = cluster_id
        meta["search_score"] = score
        meta["rec_score"] = score
        meta["query_similarity"] = float(pool_query_sims[rank_idx])
        meta["user_similarity"] = float(user_sims[rank_idx])
        meta["recency_score"] = float(recency_scores[rank_idx])
        meta["lexical_score"] = float(lexical_scores[rank_idx])
        meta["nearest_user_thread"] = int(nearest_threads[rank_idx])
        meta["age_days"] = _paper_age_days(meta.get("update_date"))
        meta["query_cluster_ids"] = query_clusters
        meta["user_cluster_ids"] = user_clusters

        selected.append(meta)
        if paper_id is not None:
            selected_ids.add(paper_id)
        per_cluster_counts[cluster_id] += 1
        if len(selected) >= n:
            break

    return selected
