"""Shared recommendation scoring helpers."""

from __future__ import annotations

from datetime import datetime
from math import exp


def paper_age_days(published_date: str) -> int | None:
    """Return bounded paper age in days, or None when the date is unavailable."""
    try:
        published = datetime.fromisoformat(published_date)
    except (ValueError, TypeError):
        return None

    age_days = (datetime.now() - published).days
    return max(0, min(age_days, 365))


def recency_score(published_date: str, halflife_days: float = 30.0) -> float:
    """Compute an exponential recency bonus in the range (0, 1]."""
    age_days = paper_age_days(published_date)
    if age_days is None:
        return 0.5

    return exp(-age_days / halflife_days)
