"""Phase 2 verification script: test SQLite user DB operations."""

from user.db import init_db, create_user, get_user, log_feedback, get_seen_ids
import numpy as np

init_db()
emb = np.random.randn(768).astype(np.float32)
emb /= np.linalg.norm(emb)
uid = create_user("Test User", emb)
print("Created:", uid)
user = get_user(uid)
print("Retrieved embedding shape:", user["embedding"].shape)
log_feedback(uid, "2401.00001", "like", cluster_id=3, score=0.87)
print("Seen IDs:", get_seen_ids(uid))
