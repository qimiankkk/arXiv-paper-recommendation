"""In-memory paper index for serving recommendations.

Loads all precomputed artifacts (embeddings, clusters, metadata) from disk
once at startup and holds them in memory for fast KNN lookups.
Designed to be cached via st.cache_resource.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class PaperIndex:
    """Holds all data needed for serving recommendations.

    Loaded once at Streamlit startup into memory via st.cache_resource.

    Attributes:
        embeddings: Shape (N, 768) float32 — all paper embeddings.
        cluster_ids: Shape (N,) int32 — cluster assignment per paper.
        centroids: Shape (k, 768) float32 — unit-norm cluster centroids.
        category_centroids: Dict mapping arXiv category string to unit-norm
            centroid vector (shape (768,) float32).
        paper_meta: Length-N list of dicts — paper metadata (id, title,
            abstract, categories, update_date, cluster_id).

    Memory note:
        At N=2M, embeddings.npy is ~5.9 GB.
        At N=50K (dev), it is ~147 MB.
    """

    def __init__(self, data_dir: str = "data") -> None:
        """Initialize the index with a data directory path.

        Args:
            data_dir: Path to the directory containing precomputed artifacts.
        """
        self.data_dir = data_dir
        self.embeddings: np.ndarray | None = None
        self.cluster_ids: np.ndarray | None = None
        self.centroids: np.ndarray | None = None
        self.category_centroids: dict[str, np.ndarray] | None = None
        self.paper_meta: list[dict] | None = None

    def load(self) -> None:
        """Load all artifacts from data_dir into memory.

        Reads the following files from self.data_dir:
            - embeddings.npy    -> self.embeddings  (N, 768) float32
            - cluster_ids.npy   -> self.cluster_ids (N,) int32
            - centroids.npy     -> self.centroids   (k, 768) float32
            - category_centroids.npy -> self.category_centroids dict
            - paper_meta.jsonl  -> self.paper_meta  list[dict]

        After loading, verifies that len(embeddings) == len(paper_meta).
        Prints memory usage: embeddings.nbytes / 1e9 GB.
        """
        data = Path(self.data_dir)

        embeddings_path = data / "embeddings.npy"
        if not embeddings_path.exists():
            return  # Data not ready; is_loaded() will return False

        self.embeddings = np.load(data / "embeddings.npy")
        self.cluster_ids = np.load(data / "cluster_ids.npy")
        self.centroids = np.load(data / "centroids.npy")
        self.category_centroids = np.load(
            data / "category_centroids.npy", allow_pickle=True
        ).item()

        self.paper_meta = []
        with open(data / "paper_meta.jsonl", "r", encoding="utf-8") as fh:
            for line in fh:
                self.paper_meta.append(json.loads(line))

        assert len(self.embeddings) == len(self.paper_meta), (
            f"Shape mismatch: embeddings {len(self.embeddings)} != "
            f"paper_meta {len(self.paper_meta)}"
        )

        print(f"PaperIndex loaded: {len(self.paper_meta)} papers, "
              f"{self.embeddings.nbytes / 1e9:.2f} GB embeddings, "
              f"{self.centroids.shape[0]} clusters, "
              f"{len(self.category_centroids)} categories")

    def is_loaded(self) -> bool:
        """Check whether the index has been successfully loaded.

        Returns:
            True if self.embeddings is not None, False otherwise.
        """
        return self.embeddings is not None
