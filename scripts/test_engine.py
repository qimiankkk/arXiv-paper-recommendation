"""Phase 3 verification script: test recommendation engine."""

from pipeline.index import PaperIndex
from recommender.engine import recommend
import numpy as np

index = PaperIndex()
index.load()

# fake user embedding
user_emb = np.random.randn(768).astype(np.float32)
user_emb /= np.linalg.norm(user_emb)

recs = recommend(user_emb, seen_ids=set(), index=index)
for i, r in enumerate(recs):
    print(f"{i+1}. [{r['id']}] {r['title'][:60]}  score={r['rec_score']:.3f}")
