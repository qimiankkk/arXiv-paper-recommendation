"""K-means clustering and category centroid computation for paper embeddings.

Provides two independent operations on the embedding matrix:
1. MiniBatchKMeans clustering (groups papers into k clusters)
2. Category centroid computation (mean embedding per arXiv category tag).

Both produce unit-norm output vectors.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np



def _normalize(x: np.ndarray) -> np.ndarray:
    """Helper for row-wise L2 normalization. Rows with near-zero norm are left as-is."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return x / norms


"""manually implements .predict() for K-means, returns index of nearest centroid for each embedding
Goes through all the embeddings and computes the dot product of the embedding with each centroid, 
    keeping the highest dot product (nearest centroid)"""
def _assign(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each row in embeddings to its nearest centroid.

    Uses the identity  ||a - b||^2 = 2 - 2 * a·b  for unit-norm vectors,
    so assignment reduces to argmax of a dot product — much faster than
    computing full pairwise distances for high-dimensional embeddings.

    Args:
        embeddings: (N, D) unit-norm float32.
        centroids:  (k, D) unit-norm float32.

    Returns:
        labels: (N,) int32 cluster indices.
    """
    # dot: (N, k)  —  chunk to avoid large intermediate allocations
    chunk = 4096
    labels = np.empty(len(embeddings), dtype=np.int32)
    for start in range(0, len(embeddings), chunk):
        end = min(start + chunk, len(embeddings))
        dots = embeddings[start:end] @ centroids.T   # (chunk, k)
        labels[start:end] = dots.argmax(axis=1).astype(np.int32)
    return labels


""""
picks centroids using the K-means++ algorithm

picks first centroid randomly, computes squared distance from each point to nearest chosen centroid 
points that are far away get higher probability of being selected as centroid next, repeats until k centroids are chosen
This spreads out initial centroids and reduces iterations needed to converge compared to random initialisation.
"""
def _kmeans_plus_plus_init(
    embeddings: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """K-means++ initialisation — spreads initial centroids out.
    Args:
        embeddings: (N, D) unit-norm float32.
        k: Number of centroids to initialise.
        rng: Numpy random Generator for reproducibility.

    Returns:
        centroids: (k, D) float32.
    """
    n = len(embeddings)
    first = rng.integers(n)
    centroids = [embeddings[first]]

    min_sq_dist = 2.0 - 2.0 * (embeddings @ centroids[0])
    min_sq_dist = np.clip(min_sq_dist, 0.0, None)  # fix: clamp negatives

    for _ in range(1, k):
        total = min_sq_dist.sum()
        if total == 0.0:
            # All points are identical to chosen centroids — pick randomly
            idx = rng.integers(n)
        else:
            probs = min_sq_dist / total
            idx = rng.choice(n, p=probs)
        new_c = embeddings[idx] 
        centroids.append(new_c)

        sq_dist = 2.0 - 2.0 * (embeddings @ new_c)
        min_sq_dist = np.minimum(min_sq_dist, sq_dist)
        min_sq_dist = np.clip(min_sq_dist, 0.0, None)  # fix: clamp after update too

    return np.array(centroids, dtype=np.float32)


"""
Training function

each run is a completely separate K-means training from scratch with a different random initialisation, keep best run (lowest inertia)

for each run:
- random seed, then initialise centroids with K-means++
- create counts, which tracks how many samples have contributed to each centroid
- for each epoch (one pass over the data within a single training attempt), shuffle data and process in mini-batches, 
assign batch points to nearest centroid, then update and renormalize each centroid
- compute inertia (sum of squared distances to nearest centroid)
- if inertia is best so far, save centroids and cluster assignments for final return

for each batch, 
- assign each point to nearest centroid using _assign()
- for each centroid, find what points were assigned to it and update that centroid toward the mean of 
those assigned points
- this is done using decaying learning rate so bigger changes at beginning, use this to update with 
weighted average between old centroid and new batch mean
- renormalize centroids to unit length after each batch update

"""
def fit_kmeans(
    embeddings: np.ndarray,
    k: int = 500,
    batch_size: int = 4096,
    n_init: int = 5,
    max_epochs: int = 20,
    tol: float = 1e-4,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster paper embeddings using from-scratch MiniBatch K-Means.

    Runs `n_init` independent initialisations and keeps the one with the
    lowest final inertia (sum of squared distances to nearest centroid).

    MiniBatch K-Means update rule (Sculley 2010):
        For each mini-batch:
          1. Assign each point to its nearest centroid.
          2. For each centroid c_j, accumulate a per-centroid learning rate:
                η_j = 1 / count_j
             and apply the update:
                c_j ← (1 - η_j) * c_j  +  η_j * mean(batch points in cluster j)
          3. Re-normalise centroids to unit length after each batch update.

    This is equivalent to an online stochastic gradient step on the K-means
    objective with a decaying per-centroid step size, giving convergence
    guarantees similar to SGD while using only O(batch_size * D) memory
    rather than the O(N * D) required by full-batch K-means.

    Args:
        embeddings: Shape (N, D), float32, unit-norm rows.
        k: Number of clusters. Default 500.
        batch_size: Mini-batch size per update step. Default 4096.
        n_init: Number of independent runs; best inertia is kept. Default 5.
        max_epochs: Maximum passes over the data per run. Default 20.
        tol: Convergence tolerance on centroid shift (L2). Default 1e-4.
        random_state: Base seed for reproducibility.

    Returns:
        Tuple of:
            cluster_ids: Shape (N,) int32 — cluster assignment for each paper.
            centroids:   Shape (k, D) float32 — unit-norm cluster centroids.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    n, d = embeddings.shape

    best_centroids: np.ndarray | None = None
    best_inertia = float("inf")

    for run in range(n_init):
        rng = np.random.default_rng(random_state + run)

        # --- Initialisation (K-means++) ---
        centroids = _kmeans_plus_plus_init(embeddings, k, rng)
        centroids = _normalize(centroids)

        # Per-centroid sample counts used for the learning-rate schedule
        counts = np.ones(k, dtype=np.float64)

        # --- MiniBatch training loop ---
        for epoch in range(max_epochs):
            perm = rng.permutation(n)
            old_centroids = centroids.copy()

            for start in range(0, n, batch_size):
                batch = embeddings[perm[start : start + batch_size]]  # (B, D)
                labels = _assign(batch, centroids)                     # (B,)

                # Accumulate updates per centroid
                for j in range(k):
                    mask = labels == j
                    if not mask.any():
                        continue
                    counts[j] += mask.sum()
                    lr = mask.sum() / counts[j]                # decaying η
                    centroids[j] = (
                        (1.0 - lr) * centroids[j]
                        + lr * batch[mask].mean(axis=0)
                    )

                centroids = _normalize(centroids)

            # Convergence check: mean centroid shift across all centroids
            shift = np.linalg.norm(centroids - old_centroids, axis=1).mean()
            if shift < tol:
                break

        # --- Evaluate inertia for this run ---
        labels = _assign(embeddings, centroids)
        # inertia = sum of squared distances (= 2 - 2*dot for unit-norm vecs)
        dots = (embeddings * centroids[labels]).sum(axis=1)  # (N,)
        inertia = float((2.0 - 2.0 * dots).sum())

        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids.copy()

    assert best_centroids is not None
    cluster_ids = _assign(embeddings, best_centroids)
    return cluster_ids, best_centroids

"""
scans paper_meta and categorizes clusters by common metadata category if possible

take all papers with a label and average their embeddings to creates one vector per category

"""
def compute_category_centroids(
    embeddings: np.ndarray,
    paper_meta: list[dict],
) -> dict[str, np.ndarray]:
    """Compute the mean embedding for each arXiv category tag.

    Args:
        embeddings: Shape (N, 768), float32, unit-norm rows.
        paper_meta: Length-N list of dicts. Each dict has a "categories" key
            containing a list of arXiv tags, e.g. ["cs.LG", "cs.CL"].

    Returns:
        Dict mapping category string to its unit-norm centroid (shape (768,), float32).
        Example: {"cs.LG": array(768,), "cs.CL": array(768,), ...}
    """
    cat_indices: dict[str, list[int]] = defaultdict(list)
    for i, meta in enumerate(paper_meta):
        for cat in meta["categories"]:
            cat_indices[cat].append(i)

    centroids: dict[str, np.ndarray] = {}
    for cat, indices in cat_indices.items():
        mean_vec = embeddings[indices].mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-8:
            mean_vec = mean_vec / norm
        centroids[cat] = mean_vec.astype(np.float32)

    return centroids