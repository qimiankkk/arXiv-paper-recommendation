"""Phase 3 verification script: test recommendation engine (v2 multi-vector)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.index import PaperIndex
from recommender.engine import recommend
import numpy as np

index = PaperIndex()
index.load()

centroids = np.random.randn(2, 768).astype(np.float32)
centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

recs = recommend(centroids, seen_ids=set(), index=index, diversity=0.5, n=5)
for i, r in enumerate(recs):
    print(f"{i+1}. [{r['id']}] {r['title'][:60]}  score={r['rec_score']:.3f}")

# Test with diversity=0.0 (more focused)
recs_focused = recommend(centroids, seen_ids=set(), index=index, diversity=0.0, n=5)
print(f"\nWith diversity=0.0: {len(recs_focused)} papers")

# Test with diversity=1.0 (very broad)
recs_broad = recommend(centroids, seen_ids=set(), index=index, diversity=1.0, n=5)
print(f"With diversity=1.0: {len(recs_broad)} papers")

# Verify k_u=1 still works (backward compatibility)
single_centroid = centroids[:1]
recs_single = recommend(single_centroid, seen_ids=set(), index=index, diversity=0.5, n=5)
print(f"\nWith k_u=1: {len(recs_single)} papers")
assert len(recs_single) <= 5
print("All tests passed!")
