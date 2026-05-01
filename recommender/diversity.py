"""Shared helpers for the diversity slider."""

from __future__ import annotations

import math


DEFAULT_DIVERSITY = 0.5


def clamp_diversity(diversity: float | None) -> float:
    """Return a finite diversity value bounded to [0.0, 1.0]."""
    if diversity is None:
        return DEFAULT_DIVERSITY
    try:
        value = float(diversity)
    except (TypeError, ValueError):
        return DEFAULT_DIVERSITY
    if not math.isfinite(value):
        return DEFAULT_DIVERSITY
    return min(1.0, max(0.0, value))
