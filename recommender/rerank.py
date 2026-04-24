"""Reranking and diversity filtering for recommendation candidates.

Takes the raw KNN candidates and applies:
1. Withdrawn paper filter — removes papers whose abstract signals withdrawal.
2. Series ordering — detects Part N papers and enforces sequential unlock:
   Part N is only shown if Part N-1 was liked/saved; skipped Part N-1
   suppresses all later parts in the same series.
3. Recency boost — newer papers get a score bonus.
4. Diversity filter — ensures selected papers come from different clusters.
5. δ-aware centroid coverage — when δ > 0.5 and k_u > 1, holds output slots
   open for uncovered user centroids before filling by score alone.
"""

from __future__ import annotations

import re
from datetime import datetime
from math import exp




# Phrases that reliably indicate a paper has been withdrawn by the authors.
_WITHDRAWAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bwithdrawn\b", re.IGNORECASE),
    re.compile(r"\bthis paper has been withdrawn\b", re.IGNORECASE),
    re.compile(r"\bthis article has been withdrawn\b", re.IGNORECASE),
    re.compile(r"\bretracted\b", re.IGNORECASE),
]

# check abstract text for withdrawn notices, returns true if the paper should be filtered out
def _is_withdrawn(meta: dict) -> bool:

    abstract = meta.get("abstract", "") or ""
    return any(p.search(abstract) for p in _WITHDRAWAL_PATTERNS)


# Series detection and ordering

# Matches common part-numbering patterns in titles
_PART_PATTERN = re.compile(
    r"[,:\s\-–—(]*\bpart\s*"      # separator + "Part"
    r"(?:(\d+)|([IVXivx]+))"      # arabic digits OR roman numerals
    r"\b",
    re.IGNORECASE,
)

_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5,
          "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}


def _roman_to_int(s: str) -> int | None:
    return _ROMAN.get(s.lower())


def _parse_series(title: str) -> tuple[str, int] | None:
    """Extract a (series_key, part_number) from a title if it contains Part N.

    The series_key is the title with the part suffix stripped and
    lowercased/normalised, so "Scaling Laws Part I" and "Scaling Laws Part II"
    share the same series_key.

    Returns None if the title has no detectable part numbering.

    Args:
        title: Paper title string.

    Returns:
        (series_key, part_number) or None.
    """
    m = _PART_PATTERN.search(title)
    if m is None:
        return None

    arabic, roman = m.group(1), m.group(2)
    if arabic:
        part_num = int(arabic)
    elif roman:
        part_num = _roman_to_int(roman)
        if part_num is None:
            return None
    else:
        return None

    # Strip the matched suffix to get the base series name
    base = title[: m.start()].strip().lower()
    # Normalise whitespace and punctuation so minor title variations still match
    base = re.sub(r"[\s\-–—:,]+", " ", base).strip()
    return base, part_num


def _filter_series(
    scored: list[tuple[float, dict, int]],
    liked_ids: set[str],
    skipped_ids: set[str],
) -> list[tuple[float, dict, int]]:
    """Enforce sequential ordering for paper series.

    Rules:
    - Part 1 of any series is always allowed through (no prerequisite).
    - Part N (N > 1) is only included if Part N-1 of the same series was
      liked or saved by the user.
    - If Part N-1 was skipped, Part N and all higher parts are suppressed.
    - Papers with no detected part numbering are unaffected.

    To apply these rules we need to know which part numbers the user has
    already seen. We scan the full candidate list to build a title→arxiv_id
    index, then check feedback signals for each prerequisite part.

    Args:
        scored: Sorted (score, meta, centroid_idx) triples — all candidates
                before diversity filtering.
        liked_ids: arxiv_ids the user liked or saved.
        skipped_ids: arxiv_ids the user skipped.

    Returns:
        Filtered list with the same ordering, series-invalid papers removed.
    """
    # Build a map: (series_key, part_number) -> arxiv_id
    # so we can look up whether a prerequisite part has been seen.
    series_map: dict[tuple[str, int], str] = {}
    for _score, meta, _ci in scored:
        title = meta.get("title", "") or ""
        parsed = _parse_series(title)
        if parsed:
            series_map[parsed] = meta.get("arxiv_id", "")

    result = []
    for score, meta, ci in scored:
        title = meta.get("title", "") or ""
        parsed = _parse_series(title)

        if parsed is None:
            # Not part of a series — always include
            result.append((score, meta, ci))
            continue

        series_key, part_num = parsed

        if part_num == 1:
            # Part 1 has no prerequisite
            result.append((score, meta, ci))
            continue

        # Check all prerequisite parts from 1 to part_num - 1
        include = True
        for prereq_num in range(1, part_num):
            prereq_arxiv_id = series_map.get((series_key, prereq_num))

            if prereq_arxiv_id is None:
                # Prerequisite isn't in the candidate pool at all.
                # Suppress: user hasn't been exposed to it yet.
                include = False
                break

            if prereq_arxiv_id in skipped_ids:
                # User explicitly skipped a prerequisite — suppress this part.
                include = False
                break

            if prereq_arxiv_id not in liked_ids:
                # Prerequisite exists but user hasn't liked/saved it yet.
                # Suppress until they do.
                include = False
                break

        if include:
            result.append((score, meta, ci))

    return result


# Recency scoring

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
        return 0.5

    age_days = (datetime.now() - published).days
    age_days = min(age_days, 365)
    age_days = max(age_days, 0)

    return exp(-age_days / halflife_days)


# Main reranking entry point

def rerank_and_select(
    candidates: list[tuple[float, dict, int]],
    k_u: int = 1,
    diversity: float = 0.5,
    recency_weight: float = 0.25,
    n: int = 5,
    liked_ids: set[str] | None = None,
    skipped_ids: set[str] | None = None,
) -> list[dict]:
    """Rerank candidates and select diverse top-n papers.

    Scoring: final_score = similarity + recency_weight * recency(date).
    Diversity: at most one paper per k-means cluster (always enforced).
    When δ > 0.5 and k_u > 1: holds slots for uncovered user centroids
    before filling remaining spots by score alone.

    Args:
        candidates: List of (sim_score, paper_meta, nearest_centroid_idx).
            Each paper_meta dict must have at minimum:
                "arxiv_id", "title", "abstract", "update_date", "cluster_id".
        k_u: Number of user centroids.
        diversity: The δ slider value, 0.0–1.0.
        recency_weight: Weight of recency bonus.
        n: Papers to select. Default 5.
        liked_ids: arxiv_ids the user liked or saved. Used for series gating.
            Pass get_seen_ids() filtered by signal, or omit to disable.
        skipped_ids: arxiv_ids the user skipped. Used to suppress later parts
            when an earlier part was skipped. Omit to disable.

    Returns:
        List of up to n paper_meta dicts, each with "rec_score" added.
    """
    liked_ids = liked_ids or set()
    skipped_ids = skipped_ids or set()

    # --- Step 1: Filter withdrawn papers ---
    candidates = [
        (sim, meta, ci)
        for sim, meta, ci in candidates
        if not _is_withdrawn(meta)
    ]

    # --- Step 2: Score and sort ---
    scored: list[tuple[float, dict, int]] = []
    for sim_score, meta, nearest_ci in candidates:
        bonus = recency_weight * recency_score(meta.get("update_date", ""))
        final = sim_score + bonus
        meta["rec_score"] = final
        scored.append((final, meta, nearest_ci))

    scored.sort(key=lambda x: x[0], reverse=True)

    # --- Step 3: Series ordering ---
    scored = _filter_series(scored, liked_ids, skipped_ids)

    # --- Step 4: Diversity selection ---
    selected: list[dict] = []
    used_clusters: set[int] = set()
    covered_centroids: set[int] = set()

    for _score, meta, nearest_ci in scored:
        cid = meta.get("cluster_id")
        if cid in used_clusters:
            continue

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