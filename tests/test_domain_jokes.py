"""Tests for the domain joke board selection logic."""

from __future__ import annotations

from datetime import date

import numpy as np

from ui.domain_jokes import JOKE_BOARDS, _pick_board_and_joke


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector."""
    return (v / np.linalg.norm(v)).astype(np.float32)


def _make_toy_embeddings(n_boards: int, dim: int = 768) -> np.ndarray:
    """Create deterministic, well-separated unit-norm board embeddings."""
    rng = np.random.RandomState(42)
    raw = rng.randn(n_boards, dim).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / np.maximum(norms, 1e-12)


# ---------------------------------------------------------------------------
# Nearest-board selection
# ---------------------------------------------------------------------------

class TestNearestBoardSelection:
    """The board whose embedding is closest to a user centroid wins."""

    def test_selects_closest_board(self) -> None:
        n = len(JOKE_BOARDS)
        joke_embs = _make_toy_embeddings(n)

        # User centroid = exact copy of board 5 → board 5 must win
        target_idx = 5
        user_centroids = joke_embs[target_idx : target_idx + 1].copy()

        result = _pick_board_and_joke(
            user_centroids, joke_embs, user_id="u1", today=date(2025, 1, 1)
        )
        assert result["label"] == JOKE_BOARDS[target_idx].label

    def test_selects_closest_among_multiple_centroids(self) -> None:
        n = len(JOKE_BOARDS)
        joke_embs = _make_toy_embeddings(n)

        # Two user centroids near boards 3 and 10; board with higher sim wins
        c1 = joke_embs[3] + 0.001 * np.random.RandomState(0).randn(768).astype(np.float32)
        c2 = joke_embs[10] + 0.001 * np.random.RandomState(1).randn(768).astype(np.float32)
        c1 = _unit(c1)
        c2 = _unit(c2)
        user_centroids = np.stack([c1, c2])

        result = _pick_board_and_joke(
            user_centroids, joke_embs, user_id="u1", today=date(2025, 6, 15)
        )
        # Should pick one of the two nearby boards
        assert result["label"] in {JOKE_BOARDS[3].label, JOKE_BOARDS[10].label}

    def test_different_centroid_picks_different_board(self) -> None:
        n = len(JOKE_BOARDS)
        joke_embs = _make_toy_embeddings(n)

        r1 = _pick_board_and_joke(
            joke_embs[0:1], joke_embs, user_id="u1", today=date(2025, 1, 1)
        )
        r2 = _pick_board_and_joke(
            joke_embs[n - 1 : n], joke_embs, user_id="u1", today=date(2025, 1, 1)
        )
        assert r1["label"] == JOKE_BOARDS[0].label
        assert r2["label"] == JOKE_BOARDS[n - 1].label
        assert r1["label"] != r2["label"]


# ---------------------------------------------------------------------------
# Stable daily joke selection
# ---------------------------------------------------------------------------

class TestStableDailySelection:
    """Same (user_id, date) always yields the same joke; changing either may change the joke."""

    def test_same_user_same_day_is_stable(self) -> None:
        joke_embs = _make_toy_embeddings(len(JOKE_BOARDS))
        centroid = joke_embs[0:1]

        results = [
            _pick_board_and_joke(centroid, joke_embs, "user-abc", date(2025, 3, 14))
            for _ in range(20)
        ]
        assert all(r == results[0] for r in results)

    def test_different_day_may_change_joke(self) -> None:
        joke_embs = _make_toy_embeddings(len(JOKE_BOARDS))
        centroid = joke_embs[0:1]

        jokes_seen = set()
        for day in range(1, 60):
            r = _pick_board_and_joke(
                centroid, joke_embs, "user-abc", date(2025, 1, day % 28 + 1)
            )
            jokes_seen.add(r["joke"])

        # Over ~28 different days we expect both jokes to appear
        assert len(jokes_seen) == 2

    def test_different_user_may_change_joke(self) -> None:
        joke_embs = _make_toy_embeddings(len(JOKE_BOARDS))
        centroid = joke_embs[0:1]
        today = date(2025, 7, 4)

        jokes_seen = set()
        for i in range(50):
            r = _pick_board_and_joke(centroid, joke_embs, f"user-{i}", today)
            jokes_seen.add(r["joke"])

        # Across many user IDs both jokes should appear
        assert len(jokes_seen) == 2


# ---------------------------------------------------------------------------
# Return shape / type
# ---------------------------------------------------------------------------

class TestReturnShape:
    def test_result_has_label_and_joke(self) -> None:
        joke_embs = _make_toy_embeddings(len(JOKE_BOARDS))
        centroid = joke_embs[0:1]
        result = _pick_board_and_joke(centroid, joke_embs, "u1", date(2025, 1, 1))
        assert "label" in result
        assert "joke" in result
        assert isinstance(result["label"], str)
        assert isinstance(result["joke"], str)

    def test_joke_is_from_selected_board(self) -> None:
        joke_embs = _make_toy_embeddings(len(JOKE_BOARDS))
        for idx in range(len(JOKE_BOARDS)):
            centroid = joke_embs[idx : idx + 1]
            result = _pick_board_and_joke(centroid, joke_embs, "u1", date(2025, 1, 1))
            board = JOKE_BOARDS[idx]
            assert result["label"] == board.label
            assert result["joke"] in board.jokes
