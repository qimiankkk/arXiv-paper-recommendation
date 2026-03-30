# ArXiv Daily

A personalized paper recommendation engine that surfaces 3 arXiv papers daily, tailored to your research interests! This web app is built with Streamlit, SPECTER2 embeddings, and a lightweight local stack — no cloud services or external databases required.

## How It Works

### The Core Idea

Your research taste is represented as a single 768-dimensional vector in the same embedding space as every paper in the arXiv corpus (~2M papers). Recommendation is geometric: the system finds papers whose embeddings are closest to yours.

### Algorithm Overview

The system has two phases: an **offline batch pipeline** that runs once (or nightly), and an **online serving pipeline** that executes per user at delivery time.

#### 1. Offline Pipeline

1. **Corpus preparation** — The arXiv dataset is loaded from its [Kaggle JSON snapshot](https://www.kaggle.com/datasets/Cornell-University/arxiv). Each record provides an ID, title, abstract, category tags (e.g. `cs.LG`), and date.
2. **Embedding** — [SPECTER2](https://huggingface.co/allenai/specter2_base), a scientific-text encoder pretrained on citation graphs, encodes each paper's *title + abstract* into a 768-dim vector. All vectors are L2-normalized to unit length, so cosine similarity reduces to a simple dot product.
3. **Clustering** — MiniBatchKMeans partitions papers into 500 clusters. This acts as a spatial index: instead of scanning all 2M papers at query time (~1.5B dot products), only the ~8,000 papers in the 2 nearest clusters are searched (~500x speedup).
4. **Category centroids** — Independently, one centroid per arXiv category label (e.g. `cs.LG`, `cs.CV`) is computed as the normalized mean of all embeddings in that category. These are used for cold-start user initialization.

#### 2. User Personalization

- **Cold start** — A new user selects topics of interest. Their initial embedding is the normalized average of the corresponding category centroids, placing them at the geometric center of their interests.
- **Feedback loop** — Each like, save, or skip updates the user embedding via Exponential Moving Average (EMA):

  ```
  u' = normalize( (1 - 0.15) * u  +  0.15 * w * e_paper )
  ```

  where `w` = +1.5 (save), +1.0 (like), or -0.3 (skip). The small learning rate (0.15) ensures no single paper dramatically redirects the taste vector.

#### 3. Online Serving

1. **Cluster selection** — Dot-product the user vector against all 500 cluster centroids; pick the top 2.
2. **KNN retrieval** — Brute-force dot-product search within those clusters, filtering out previously seen papers. Keep the top 40 candidates.
3. **Re-ranking** — Adjust scores with a recency bonus: `score = similarity + 0.25 * exp(-age_days / 30)`. Newer papers get a boost.
4. **Diversity filter** — Iterate candidates by score; select a paper only if its cluster hasn't already contributed one. Stop at 3 papers.

Total serving cost: <1 ms on CPU per user.

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| N | ~2,000,000 | Papers in the corpus |
| d | 768 | Embedding dimension (SPECTER2) |
| k | 500 | Number of k-means clusters |
| alpha | 0.15 | EMA learning rate |
| M | 40 | KNN candidate pool size |
| n | 3 | Papers served per day |

## Tech Stack

- **Python 3.11+**
- **Streamlit** — UI and app framework
- **sentence-transformers** — SPECTER2 embeddings
- **scikit-learn** — MiniBatchKMeans clustering
- **NumPy** — KNN search, EMA updates, all linear algebra
- **SQLite** — User profiles and feedback log (via Python `sqlite3`)
- **kagglehub** — Dataset download

No Docker. No external vector DB. No cloud services. Everything runs locally with a SQLite file next to the app.

## Project Structure

```
arXiv-paper-recommendation/
├── app.py                       # Streamlit entry point
├── README.md
├── requirements.txt
├── .streamlit/config.toml       # Theme + server config
│
├── docs/                        # Algorithm documentations
│   ├── arxiv_rec_algorithm_description.html
│   └── final_proj_alg_v1.pdf
│
├── pipeline/
│   ├── __init__.py
│   ├── embed.py                 # SPECTER2 embedding wrapper
│   ├── cluster.py               # k-means + category centroids
│   ├── index.py                 # In-memory paper index for serving
│   └── offline.py               # Orchestrates embed → cluster → save
│
├── user/
│   ├── __init__.py
│   ├── db.py                    # SQLite schema + CRUD
│   ├── profile.py               # Cold-start init, EMA update
│   └── session.py               # Streamlit session state helpers
│
├── recommender/
│   ├── __init__.py
│   ├── retrieve.py              # Cluster selection + KNN
│   ├── rerank.py                # Recency boost + diversity filter
│   └── engine.py                # Top-level recommend() function
│
├── ui/
│   ├── __init__.py
│   ├── components.py            # Reusable Streamlit widgets
│   ├── onboarding.py            # Topic selection page
│   └── daily_feed.py            # 3-paper card view
│
├── data/                        # Local artifacts (created at runtime)
│   ├── .gitkeep
│   ├── embeddings.npy           # N x 768 float32
│   ├── cluster_ids.npy          # N, int32
│   ├── centroids.npy            # 500 x 768 float32
│   ├── category_centroids.npy   # Dict of category → vector
│   ├── paper_meta.jsonl         # One JSON object per line
│   └── arxiv_rec.db             # SQLite user database
│
└── scripts/
    ├── run_offline_pipeline.py  # CLI for offline pipeline generation
    ├── reset_db.py              # Reset local SQLite test data
    ├── test_db.py               # Quick DB sanity checks
    └── test_engine.py           # Quick recommendation sanity checks
```

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the offline pipeline

This downloads the arXiv dataset, embeds papers, and clusters them. Use `--limit` for faster dev iterations:

```bash
# Development (fast, ~5 min on CPU)
python scripts/run_offline_pipeline.py --limit 50000

# Full dataset (~4 hours on GPU, overnight on CPU)
python scripts/run_offline_pipeline.py
```

### 3. Launch the app

```bash
streamlit run app.py
```

Visit `http://localhost:8501`, pick your topics, and start reading.

### Reset test data (for testing use)

To clear local user/test data and recreate an empty DB schema:

```bash
python scripts/reset_db.py
```
