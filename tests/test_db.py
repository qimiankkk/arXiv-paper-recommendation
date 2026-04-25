"""Phase 2 verification script: test SQLite user DB operations (v2 schema)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from user.db import init_db, create_user, get_user, log_feedback, get_seen_ids
import numpy as np

init_db()

centroids = np.random.randn(2, 768).astype(np.float32)
centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

uid = create_user("Test User", centroids, k_u=2, diversity=0.7)
print("Created:", uid)

user = get_user(uid)
print("Centroids shape:", user["centroids"].shape)
print("k_u:", user["k_u"], "diversity:", user["diversity"])

log_feedback(uid, "2401.00001", "like", cluster_id=3, score=0.87)
print("Seen IDs:", get_seen_ids(uid))

assert user["centroids"].shape == (2, 768)
assert user["k_u"] == 2
assert abs(user["diversity"] - 0.7) < 1e-6
assert np.allclose(np.linalg.norm(user["centroids"], axis=1), 1.0)
print("All tests passed!")
