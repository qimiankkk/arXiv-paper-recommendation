"""SQLite database schema and CRUD operations for user profiles and feedback.

Two tables:
    users    — stores user profile with serialized multi-vector centroids blob.
    feedback — logs every like/save/skip interaction for analytics and seen_ids.

No ORM; uses plain sqlite3. All writes are synchronous (Streamlit is
single-user per session, so no threading is needed).

Centroids serialization:
    Store:  centroids.astype(np.float32).tobytes()   # shape (k_u, 768) flattened
    Load:   np.frombuffer(blob, dtype=np.float32).copy().reshape(k_u, 768)
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone

import numpy as np

DB_PATH = "data/arxiv_rec.db"


def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Return a connection to the SQLite database."""
    return sqlite3.connect(db_path)


def init_db(db_path: str = DB_PATH) -> None:
    """Create the users and feedback tables if they do not exist.

    Args:
        db_path: Path to the SQLite database file.
    """
    global DB_PATH
    DB_PATH = db_path

    conn = _connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id      TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            centroids    BLOB NOT NULL,
            k_u          INTEGER NOT NULL DEFAULT 1,
            diversity    REAL NOT NULL DEFAULT 0.5,
            created_at   TEXT NOT NULL,
            last_active  TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    TEXT NOT NULL,
            arxiv_id   TEXT NOT NULL,
            signal     TEXT NOT NULL,
            cluster_id INTEGER NOT NULL,
            centroid_idx INTEGER,
            score      REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cols = {
        row[1] for row in conn.execute("PRAGMA table_info(feedback)").fetchall()
    }
    if "centroid_idx" not in cols:
        conn.execute("ALTER TABLE feedback ADD COLUMN centroid_idx INTEGER")
    conn.commit()
    conn.close()


def create_user(
    display_name: str,
    centroids: np.ndarray,
    k_u: int,
    diversity: float = 0.5,
) -> str:
    """Create a new user record in the database.

    Args:
        display_name: The user's display name.
        centroids: The user's initial centroids, shape (k_u, 768), float32, unit-norm rows.
        k_u: Number of research threads (1–3).
        diversity: The diversity slider value, 0.0–1.0.

    Returns:
        The generated user_id (UUID string).
    """
    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    blob = centroids.astype(np.float32).tobytes()

    conn = _connect()
    conn.execute(
        "INSERT INTO users (user_id, display_name, centroids, k_u, diversity, "
        "created_at, last_active) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, display_name, blob, k_u, diversity, now, now),
    )
    conn.commit()
    conn.close()
    return user_id


def get_user(user_id: str) -> dict | None:
    """Retrieve a user record by user_id.

    Args:
        user_id: The UUID string of the user.

    Returns:
        Dict with keys: user_id, display_name, centroids (np.ndarray shape (k_u, 768)),
        k_u, diversity, created_at, last_active. Returns None if user not found.
    """
    conn = _connect()
    row = conn.execute(
        "SELECT user_id, display_name, centroids, k_u, diversity, "
        "created_at, last_active FROM users WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    conn.close()

    if row is None:
        return None

    k_u = row[3]
    return {
        "user_id": row[0],
        "display_name": row[1],
        "centroids": np.frombuffer(row[2], dtype=np.float32).copy().reshape(k_u, 768),
        "k_u": k_u,
        "diversity": row[4],
        "created_at": row[5],
        "last_active": row[6],
    }


def update_centroids(user_id: str, centroids: np.ndarray) -> None:
    """Update a user's centroids in the database.

    Args:
        user_id: The UUID string of the user.
        centroids: The new centroids, shape (k_u, 768), float32, unit-norm rows.
    """
    now = datetime.now(timezone.utc).isoformat()
    blob = centroids.astype(np.float32).tobytes()

    conn = _connect()
    conn.execute(
        "UPDATE users SET centroids = ?, last_active = ? WHERE user_id = ?",
        (blob, now, user_id),
    )
    conn.commit()
    conn.close()


def log_feedback(
    user_id: str,
    arxiv_id: str,
    signal: str,
    cluster_id: int,
    score: float,
    centroid_idx: int | None = None,
) -> None:
    """Log a user feedback event (like/save/skip) to the feedback table.

    Args:
        user_id: The UUID string of the user.
        arxiv_id: The arXiv paper ID.
        signal: One of "like", "save", "skip".
        cluster_id: The cluster ID of the paper.
        centroid_idx: The matched user centroid index for this recommendation.
        score: The recommendation score at the time of serving.
    """
    now = datetime.now(timezone.utc).isoformat()

    conn = _connect()
    conn.execute(
        "INSERT INTO feedback (user_id, arxiv_id, signal, cluster_id, centroid_idx, score, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, arxiv_id, signal, cluster_id, centroid_idx, score, now),
    )
    conn.commit()
    conn.close()


def get_seen_ids(user_id: str) -> set[str]:
    """Get the set of all arXiv paper IDs this user has interacted with.

    Args:
        user_id: The UUID string of the user.

    Returns:
        Set of arxiv_id strings from all feedback rows for this user.
    """
    conn = _connect()
    rows = conn.execute(
        "SELECT arxiv_id FROM feedback WHERE user_id = ?",
        (user_id,),
    ).fetchall()
    conn.close()

    return {row[0] for row in rows}


def get_like_counts_by_centroid(user_id: str, k_u: int) -> list[int]:
    """Return like-counts per user centroid index.

    Args:
        user_id: The UUID string of the user.
        k_u: Number of user centroids.

    Returns:
        A length-k_u list where item i is number of "like" events
        with centroid_idx=i.
    """
    counts = [0 for _ in range(max(0, k_u))]
    if k_u <= 0:
        return counts

    conn = _connect()
    rows = conn.execute(
        "SELECT centroid_idx, COUNT(*) FROM feedback "
        "WHERE user_id = ? AND signal = 'like' AND centroid_idx IS NOT NULL "
        "GROUP BY centroid_idx",
        (user_id,),
    ).fetchall()
    conn.close()

    for centroid_idx, cnt in rows:
        if centroid_idx is None:
            continue
        if 0 <= int(centroid_idx) < k_u:
            counts[int(centroid_idx)] = int(cnt)
    return counts
