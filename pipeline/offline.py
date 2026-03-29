"""Offline pipeline orchestrator: download → embed → cluster → save.

Runs the full data preparation pipeline. Meant to be invoked via
scripts/run_offline_pipeline.py (CLI) or programmatically.
Produces all artifacts in data_dir that the serving layer (PaperIndex) needs.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import kagglehub
import numpy as np

from pipeline.embed import EmbeddingModel
from pipeline.cluster import fit_kmeans, compute_category_centroids


def run(limit: int | None = None, k: int = 500, data_dir: str = "data") -> None:
    """Execute the full offline pipeline.

    Args:
        limit: Maximum number of papers to process. None = all (~2M).
            Use 10000-50000 during development for faster iteration.
        k: Number of k-means clusters. Default 500.
        data_dir: Output directory for all generated artifacts.
    """
    os.makedirs(data_dir, exist_ok=True)

    # ── Step 1: Download dataset ─────────────────────────────────────────
    t0 = time.time()
    print("Step 1/6: Downloading ArXiv dataset...")
    dataset_path = kagglehub.dataset_download("Cornell-University/arxiv")
    # Find the .json file inside the downloaded path
    json_file = None
    for root, _dirs, files in os.walk(dataset_path):
        for f in files:
            if f.endswith(".json"):
                json_file = os.path.join(root, f)
                break
        if json_file:
            break
    if json_file is None:
        raise FileNotFoundError(f"No .json file found in {dataset_path}")
    print(f"  Dataset at: {json_file}  ({time.time() - t0:.1f}s)")

    # ── Step 2: Load papers from JSON ────────────────────────────────────
    t1 = time.time()
    print("Step 2/6: Loading papers from JSON...")
    papers: list[dict] = []
    with open(json_file, "r", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            title = record.get("title", "").strip()
            abstract = record.get("abstract", "").strip()
            if not title or not abstract:
                continue
            papers.append({
                "id": record["id"],
                "title": title,
                "abstract": abstract,
                "categories": record.get("categories", "").split(),
                "update_date": record.get("update_date", ""),
            })
            if limit is not None and len(papers) >= limit:
                break
    print(f"  Loaded {len(papers)} papers  ({time.time() - t1:.1f}s)")

    # ── Step 3: Embed ────────────────────────────────────────────────────
    t2 = time.time()
    print("Step 3/6: Embedding papers with SPECTER2...")
    model = EmbeddingModel()
    embeddings = model.embed_papers(papers)
    print(f"  Embeddings shape: {embeddings.shape}  ({time.time() - t2:.1f}s)")

    # ── Step 4: Cluster + category centroids ─────────────────────────────
    t3 = time.time()
    print("Step 4/6: Clustering...")
    cluster_ids, centroids = fit_kmeans(embeddings, k)
    print(f"  Clusters: {centroids.shape[0]}, centroids shape: {centroids.shape}")

    print("  Computing category centroids...")
    category_centroids = compute_category_centroids(embeddings, papers)
    print(f"  Category centroids: {len(category_centroids)} categories  ({time.time() - t3:.1f}s)")

    # ── Step 5: Attach cluster_id to metadata ────────────────────────────
    t4 = time.time()
    print("Step 5/6: Attaching cluster IDs to metadata...")
    for i, cid in enumerate(cluster_ids):
        papers[i]["cluster_id"] = int(cid)
    print(f"  Done  ({time.time() - t4:.1f}s)")

    # ── Step 6: Save artifacts ───────────────────────────────────────────
    t5 = time.time()
    print("Step 6/6: Saving artifacts...")
    data = Path(data_dir)

    np.save(data / "embeddings.npy", embeddings)
    np.save(data / "cluster_ids.npy", cluster_ids)
    np.save(data / "centroids.npy", centroids)
    np.save(data / "category_centroids.npy", category_centroids, allow_pickle=True)

    with open(data / "paper_meta.jsonl", "w", encoding="utf-8") as fh:
        for paper in papers:
            fh.write(json.dumps(paper) + "\n")

    print(f"  Saved to {data_dir}/  ({time.time() - t5:.1f}s)")
    print(f"\nPipeline complete. Total time: {time.time() - t0:.1f}s")
    print(f"  embeddings: {embeddings.shape}  ({embeddings.nbytes / 1e6:.1f} MB)")
    print(f"  centroids:  {centroids.shape}")
    print(f"  categories: {len(category_centroids)}")
    print(f"  papers:     {len(papers)}")
