# ArXiv Daily — Design Walkthrough

This document explains **why** the codebase is structured the way it is. It walks through every module, the data flow between them, and the reasoning behind each design decision. By the end you should be able to trace a recommendation from raw arXiv JSON all the way to a user's screen.

---

## The Big Picture

The system has two completely separate execution paths that share a common data format:

```
┌─────────────────────────────────────────────────┐
│              OFFLINE  (run once)                │
│  arXiv JSON → embed → cluster → save to disk   │
└──────────────────────┬──────────────────────────┘
                       │  .npy files + .jsonl
                       ▼
┌─────────────────────────────────────────────────┐
│              ONLINE  (Streamlit app)            │
│  load index → user onboards → recommend → feed  │
└─────────────────────────────────────────────────┘
```

The offline pipeline produces five artifact files. The online app loads them into memory and never touches the raw data again. This separation means:

- The expensive embedding step (~hours on CPU) happens once.
- The app starts in seconds — it just memory-maps numpy arrays.
- You can re-run the pipeline on a schedule without touching the app.

---

## Module Map

```
pipeline/          Offline data processing — no Streamlit dependency
├── embed.py       SPECTER2 model wrapper
├── cluster.py     K-means + category centroids
├── index.py       In-memory index for serving (loaded once at app start)
└── offline.py     Orchestrator that wires embed → cluster → save

user/              User state management
├── db.py          SQLite schema + CRUD (no ORM)
├── profile.py     Cold-start init + EMA feedback math
└── session.py     Bridges SQLite ↔ Streamlit session state

recommender/       Online recommendation logic
├── retrieve.py    Cluster selection + KNN within clusters
├── rerank.py      Recency boost + diversity filter
└── engine.py      Top-level recommend() function

ui/                Streamlit pages and widgets
├── components.py  Reusable widgets (paper card, topic selector)
├── onboarding.py  New user flow
└── daily_feed.py  Main paper feed with feedback handling

app.py             Entry point — routes between onboarding and feed
```

There are no cross-dependencies between `pipeline/`, `user/`, and `recommender/` except through `PaperIndex`, which is the read-only data contract between offline and online.

---

## The Shared Embedding Space

Every design decision flows from one core idea: **users and papers live in the same 768-dimensional vector space**.

SPECTER2 (a BERT variant trained on scientific citation graphs) encodes each paper's title+abstract into a 768-dim vector. We normalize every vector to unit length. This gives us a powerful invariant:

> **dot product = cosine similarity** everywhere, always.

There is no cosine division anywhere in the codebase. If you ever see one, it means a normalization was missed. This invariant holds for:

- Paper embeddings (normalized at encoding time in `embed.py`)
- Cluster centroids (normalized after k-means in `cluster.py`)
- Category centroids (normalized after averaging in `cluster.py`)
- User embeddings (normalized at init in `profile.py`, re-normalized after every EMA update)

---

## Module-by-Module Walkthrough

### `pipeline/embed.py` — The Encoder

**Why this structure?** SPECTER2 has a base model and task-specific adapters. We use the `proximity` adapter because our task is nearest-neighbor retrieval. The `adapters` library lets us load this on top of the base model without merging weights.

**Why `title + [SEP] + abstract`?** This is the input format SPECTER2 was trained on. The `[SEP]` token tells the model where the title ends and the abstract begins, which matters because the model weights learned to treat these segments differently. Earlier versions of this code used `title. abstract` (period separator), but the `[SEP]` token is what the model expects.

**Why batch size 64?** This balances GPU memory usage against throughput. The tokenizer pads all sequences in a batch to the longest one (max 512 tokens), so larger batches waste more memory on padding. 64 is a sweet spot for BERT-base models.

**Why normalize inside `embed_batch`?** Normalization happens as early as possible — right after the model produces the CLS vector. This way, no downstream code ever has to worry about whether a vector is normalized. The `embed_papers` method returns a guaranteed unit-norm matrix.

---

### `pipeline/cluster.py` — Spatial Indexing

**Why k-means?** The real purpose of clustering is **not** topic discovery. It's a spatial index that makes KNN search 500x faster. With 2M papers, brute-force KNN requires 2M dot products per query. With k=500 clusters (~4000 papers each), we:

1. Compute 500 dot products to find the best 2 clusters.
2. Compute ~8000 dot products within those clusters.

Total: ~8500 operations instead of 2M.

**Why MiniBatchKMeans?** Standard KMeans loads all data into memory for each iteration. MiniBatchKMeans processes random 4096-sample batches, which converges faster on large datasets and uses less memory.

**Why `compute_category_centroids` is separate?** Category centroids and k-means solve different problems:

- **K-means centroids** are anonymous. Cluster 247 doesn't have a human-readable label. They're for fast retrieval.
- **Category centroids** are labeled (e.g., `cs.LG`). They're used exclusively for cold-start user initialization — we need to map a user's topic selections (which are arXiv category codes) to a starting position in embedding space.

These two sets of centroids are computed from the same embeddings but stored separately and used in completely different contexts.

---

### `pipeline/index.py` — The Data Contract

**Why a class?** `PaperIndex` is the single point of contact between the offline pipeline and the online serving layer. It loads five files and exposes them as typed attributes. This gives us one place to:

- Verify data consistency (do embeddings and metadata have the same length?)
- Report memory usage
- Gate the app on data availability (`is_loaded()`)

**Why `st.cache_resource`?** Streamlit re-runs the entire script on every user interaction. Without caching, the app would reload 6 GB of embeddings on every button click. `st.cache_resource` ensures the index is loaded exactly once per process and shared across all reruns and sessions.

**Why not a database for embeddings?** A numpy `.npy` file can be memory-mapped directly. There's no serialization overhead, no query parser, no connection pool. For a single-machine app serving a single user, this is simpler and faster than any vector database.

---

### `pipeline/offline.py` — The Orchestrator

**Why a single `run()` function?** The offline pipeline is a linear sequence: download → parse → embed → cluster → save. There's no branching, no parallelism (except what numpy/sklearn do internally), and no partial execution needed. A single function with print-based progress is the simplest thing that works.

**Why `--limit`?** At 2M papers, the full pipeline takes hours on CPU. During development, `--limit 50000` produces enough data to test the entire app in ~5 minutes. The flag lets you iterate on app code without waiting for a full pipeline run.

**Why JSONL for metadata?** The metadata file (`paper_meta.jsonl`) stores one JSON object per line. This format is:

- Streamable (you can process it line by line without loading all 2M records into memory)
- Human-readable (open it in any text editor)
- Append-friendly (for future incremental updates)

---

### `user/db.py` — Persistence

**Why SQLite?** The app runs on a single machine. SQLite requires zero configuration, needs no separate server process, and stores everything in one file next to the app. For a single-user-per-session Streamlit app, it's the right tool.

**Why two tables?** The `users` table stores the current state (who the user is, their current embedding). The `feedback` table is an append-only log of every interaction. This separation lets us:

- Reconstruct a user's embedding from scratch if needed (replay all feedback)
- Analyze engagement patterns (which clusters get the most likes?)
- Compute `seen_ids` (a cumulative set of all papers ever served)

**Why store embeddings as BLOBs?** A 768-dim float32 array is 3072 bytes. Storing it as a binary blob (via `tobytes()` / `frombuffer()`) avoids the overhead of 768 separate columns or a JSON array. The trade-off is that you can't query individual dimensions in SQL — but we never need to.

**Why a global `DB_PATH`?** `init_db()` sets the module-level `DB_PATH` once at startup. All other functions use `_connect()` which reads this global. This avoids passing `db_path` through every function call while keeping the path configurable for testing.

---

### `user/profile.py` — The Math

**Why EMA?** Exponential Moving Average is the simplest feedback mechanism that has the properties we need:

- **Recency bias**: Recent feedback has more influence than old feedback (the "exponential" part).
- **Stability**: No single interaction can dramatically change the user vector (controlled by alpha=0.15).
- **No history storage**: We only need the current embedding, not the full feedback history.

The update formula `u' = normalize((1-α)u + α·w·e_paper)` does three things:

1. Blends the user's current position with the paper's position.
2. Scales the paper's influence by the feedback weight (`w`).
3. Re-normalizes to maintain the unit-length invariant.

**Why asymmetric weights?** Skips get weight -0.3, but likes get +1.0. This is intentional:

- A skip means "not interesting right now" — it's weak information. The user might skip a great paper because they're busy.
- A like means "this is what I want" — it's strong information.
- A save means "I want to read this carefully" — even stronger.

If skips had weight -1.0, a few accidental skips could push the user away from their actual interests.

**Why the norm guard?** The formula `(1-α)u + α·w·e_paper` with w=-0.3 could theoretically produce a near-zero vector if the user embedding and paper embedding are very similar and w is negative. The `if norm < 1e-8: return user_embedding` guard prevents a division-by-zero. In practice with α=0.15 this can't happen, but the guard costs nothing and prevents a crash.

---

### `user/session.py` — Bridging Two Worlds

**Why does this module exist?** Streamlit and SQLite have fundamentally different state models:

- **Streamlit** stores state in `st.session_state` — a per-tab, in-memory dict that survives reruns but not page refreshes.
- **SQLite** stores state on disk — survives everything but is slower to access.

`session.py` keeps these in sync. The user embedding lives in both places: session state for fast access during recommendation, and SQLite for persistence across sessions. When feedback updates the embedding, both are updated together (in `daily_feed.py`'s `_handle_feedback`).

---

### `recommender/retrieve.py` — Finding Candidates

**Why two-stage retrieval?** Searching all 2M papers is too slow. Searching just one cluster might miss relevant papers near cluster boundaries. The two-stage approach (find 2 best clusters → KNN within them) balances speed and recall:

- **Stage 1**: 500 dot products to find 2 clusters. Negligible cost.
- **Stage 2**: ~8000 dot products within those clusters. Still fast.

**Why top-2 clusters?** A user interested in "NLP" might have their embedding between the "computational linguistics" cluster and the "language models" cluster. Top-1 would miss half their interests. Top-2 catches papers at the boundaries. Top-3+ has diminishing returns and increases latency.

**Why filter `seen_ids` here?** Filtering happens inside the KNN loop, not after. This is important: if we first took the top-40 by similarity and then removed seen papers, we might end up with fewer than 40 candidates. By filtering during iteration, we always try to fill the full pool.

---

### `recommender/rerank.py` — Quality Polish

**Why recency boost?** Pure similarity ranking ignores time. A paper from 2019 about transformers might be extremely similar to a user interested in LLMs, but a 2024 paper about the same topic is more actionable. The recency score `exp(-age/30)` gives full marks to papers from the last week and smoothly decays to near-zero for papers older than a few months.

**Why clamp at 365 days?** Without clamping, a 10-year-old paper would get a recency score of ~0. The exponential would underflow. Clamping at 365 days means old papers get a small but nonzero recency score, keeping the math clean.

**Why a diversity filter?** Without it, all 3 recommended papers could come from the same dense cluster (e.g., three variations on "attention mechanisms"). The diversity filter — select a paper only if its cluster hasn't contributed one yet — guarantees topical spread. This is a greedy approach: iterate by score, skip duplicated clusters. It's O(M) and simple to reason about.

---

### `recommender/engine.py` — The Orchestrator

**Why the all-cluster fallback?** If the user has seen most papers in their top-2 clusters (possible with a small dev dataset), the normal pipeline returns fewer than 3 papers. The fallback searches ALL clusters, which is slower but guarantees we find unseen papers if any exist. This only triggers on edge cases.

---

### `ui/components.py` — Reusable Widgets

**Why `TOPIC_LABELS`?** arXiv category codes like `cs.LG` are not user-friendly. The dict maps human-readable names to codes. The `topic_selector` function filters this dict to only show topics that actually exist in the loaded corpus — if a dev dataset doesn't have any `q-fin.CP` papers, that option won't appear.

**Why `responded` state in `paper_card`?** Streamlit reruns the entire script on every interaction. Without tracking which papers have been responded to, a user could click "Like" → script reruns → button is enabled again → click "Like" again → duplicate feedback logged. The `responded` set in session state prevents this.

---

### `ui/onboarding.py` — First Impression

**Why is validation inline?** The "Start reading" button handler validates name and topic selection before creating the user. If validation fails, it shows an error and returns — the page stays on onboarding. If it passes, it creates the user, sets session state, and calls `st.rerun()` to immediately show the daily feed.

**Why `st.rerun()`?** After creating the user, we need Streamlit to re-execute the script from the top. On this second run, `is_onboarded()` returns True, so `app.py` routes to `render_daily_feed` instead of `render_onboarding`. Without the rerun, the user would see both pages rendered in sequence.

---

### `ui/daily_feed.py` — The Main Loop

**Why cache recommendations in session state?** `recommend()` involves thousands of dot products. If we called it on every rerun (which happens on every button click), the user would see a loading spinner every time they interact. By caching in `st.session_state["todays_recs"]`, recommendations are computed once and reused until the session ends.

**Why `st.rerun()` after feedback?** When a user clicks "Like", the `_handle_feedback` function updates the embedding and adds the paper to `responded`. But Streamlit has already rendered the buttons for this run. The rerun forces a fresh render where the buttons appear disabled.

**Why look up `paper_idx` by linear scan?** To apply the EMA update, we need the paper's embedding vector. We find it by scanning `index.paper_meta` for a matching ID. This is O(N) in the worst case, but it only runs on user click (3 times per session at most), so the latency is imperceptible.

---

### `app.py` — The Router

The entry point is deliberately thin — 20 lines of logic. It does four things:

1. **Init DB** — ensures tables exist.
2. **Init session** — sets default state for new visitors.
3. **Load index** — cached, so it's fast after the first load.
4. **Route** — onboarding or daily feed, based on session state.

This keeps the routing logic obvious and puts all page-specific complexity inside the `ui/` modules.

---

## Data Flow: End to End

### Offline (run once)

```
arXiv JSON  ──→  EmbeddingModel.embed_papers()  ──→  embeddings.npy (N×768)
                                                 ──→  paper_meta.jsonl
embeddings  ──→  fit_kmeans()                    ──→  cluster_ids.npy, centroids.npy
embeddings  ──→  compute_category_centroids()    ──→  category_centroids.npy
```

### Online (per user session)

```
User selects topics  ──→  init_embedding_from_topics()  ──→  user embedding (768,)
                                                         ──→  saved to SQLite + session

User opens app  ──→  recommend(user_emb, seen_ids, index)
                     ├─ find_nearest_clusters() → 2 cluster IDs
                     ├─ knn_in_clusters() → 40 candidates
                     └─ rerank_and_select() → 3 papers

User clicks Like  ──→  log_feedback() → SQLite
                  ──→  apply_feedback() → new embedding
                  ──→  update_embedding() → SQLite
                  ──→  save_embedding_to_session() → session state
```

---

## Key Design Decisions

| Decision | Alternative considered | Why this way |
|---|---|---|
| numpy arrays, not a vector DB | FAISS, Pinecone, ChromaDB | Single machine, no server, simpler deployment. 6 GB in RAM is fine. |
| SQLite, not Postgres | PostgreSQL, Redis | Zero config, single file, no server process. Matches the "run locally" constraint. |
| K-means for retrieval, not topics | LSH, HNSW, exact search | Simple, deterministic, and 500x speedup is sufficient for 2M papers. |
| EMA for personalization, not collaborative filtering | Matrix factorization, neural CF | EMA needs no other users. It works from day one with a single user. |
| SPECTER2 + adapters, not sentence-transformers | sentence-transformers wrapping | Matches the model's documented usage: base model + task-specific adapter for proximity/retrieval. |
| Streamlit, not Flask/React | Full-stack web app | Rapid prototyping, built-in state management, zero frontend build step. |
| Unit-norm everywhere | Normalize only at query time | Eliminates an entire class of bugs. Dot product = cosine similarity, always. |
