from pathlib import Path
from uuid import uuid4

import numpy as np

from user.db import create_user, get_seen_ids, get_user, init_db, log_feedback


def _unit_centroids(k_u: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centroids = rng.normal(size=(k_u, 768)).astype(np.float32)
    return centroids / np.linalg.norm(centroids, axis=1, keepdims=True)


def test_user_db_operations_use_explicit_temp_database():
    db_file = Path("data") / f"test_user_{uuid4().hex}.db"
    db_file.parent.mkdir(parents=True, exist_ok=True)
    db_path = str(db_file)
    init_db(db_path)

    try:
        centroids = _unit_centroids(2)
        uid = create_user("Test User", centroids, k_u=2, diversity=0.7)

        user = get_user(uid)
        assert user is not None
        assert user["centroids"].shape == (2, 768)
        assert user["k_u"] == 2
        assert abs(user["diversity"] - 0.7) < 1e-6
        assert np.allclose(np.linalg.norm(user["centroids"], axis=1), 1.0)

        log_feedback(uid, "2401.00001", "like", cluster_id=3, score=0.87)
        assert get_seen_ids(uid) == {"2401.00001"}
    finally:
        db_file.unlink(missing_ok=True)
