# Diversity Index Score Design

## Goal

The diversity index, `delta`, controls how broad a user's recommendation feed
should be. The current retrieval design already uses `delta` well for search
breadth:

```text
cluster_budget(delta) = ceil(base + delta * multiplier)
```

With the current constants:

```text
base = DAILY_CLUSTER_BUDGET_BASE = 4
multiplier = DAILY_CLUSTER_BUDGET_DIVERSITY_MULTIPLIER = 8
cluster_budget(delta) = ceil(4 + 8 * delta)
```

This design keeps that retrieval-breadth formula unchanged. The proposed change
is limited to the final recommendation selection score so that `delta` also
controls how strongly the feed covers the user's multiple research centroids.

This design intentionally does not penalize a candidate paper for being similar
to papers already selected. It avoids an MMR-style selected-paper redundancy
term and instead uses user-centroid coverage and cluster saturation.

## Score Formula

For a candidate paper `p`, selected feed prefix `S`, and user profile centroids
`U = {u_1, ..., u_k}`, compute:

```text
score(p | S, U, delta) =
    relevance(p, U)
  + alpha * recency(p)
  + delta * beta * centroid_coverage(p | S, U)
  - delta * eta * cluster_saturation(p | S)
```

Expanded:

```text
score(p | S, U, delta) =
    max_i cosine(e_p, u_i)
  + alpha * exp(-age_days(p) / tau)
  + delta * beta * (1 / (1 + thread_count_S(nearest_thread(p))))
  - delta * eta * (cluster_count_S(cluster_id(p)) / max_per_cluster)
```

Recommended initial weights:

```text
alpha = 0.25
beta  = 0.10
eta   = 0.05
tau   = 30 days
```

These weights should be tuned empirically. They are deliberately small because
semantic relevance should remain the dominant signal.

## Variable Definitions

### Candidate Paper

```text
p
```

A paper candidate returned by retrieval. In the current code this is represented
as:

```text
(sim_score, paper_meta, nearest_centroid_idx)
```

from `knn_in_clusters(...)`.

### Selected Prefix

```text
S
```

The list of papers already selected into the feed during greedy selection.
Because the diversity terms depend on `S`, candidate scores are recomputed after
each selected paper.

### User Centroids

```text
U = {u_1, ..., u_k}
```

The user's research-interest vectors. `k` is `k_u` in the codebase. These
vectors are expected to be unit-normalized and live in the same embedding space
as paper embeddings.

### Paper Embedding

```text
e_p
```

The unit-normalized embedding for paper `p`.

The current candidate tuple does not include `e_p`, but the final score formula
does not require selected-paper similarity, so the selector can continue to use
the raw similarity and nearest centroid already returned by `knn_in_clusters`.
If later diagnostics need exact per-centroid scores, the candidate payload can be
expanded.

### Diversity Index

```text
delta
```

The user's diversity preference, bounded to:

```text
0.0 <= delta <= 1.0
```

Interpretation:

```text
delta = 0.0 -> focused ranking, mostly relevance + recency
delta = 1.0 -> stronger coverage across user centroids and clusters
```

Serving code should clamp the stored value before using it:

```text
delta_clamped = min(1.0, max(0.0, delta))
```

### Relevance

```text
relevance(p, U) = max_i cosine(e_p, u_i)
```

This is the best semantic match between paper `p` and any user centroid. Since
embeddings are unit-normalized, cosine similarity is computed as a dot product.

In the current code this is already computed as `sim_score` by
`knn_in_clusters(...)`.

### Nearest User Thread

```text
nearest_thread(p) = argmax_i cosine(e_p, u_i)
```

This is the user centroid that best matches paper `p`.

In the current code this is already computed as `nearest_centroid_idx` by
`knn_in_clusters(...)`.

### Recency

```text
recency(p) = exp(-age_days(p) / tau)
```

Where:

```text
age_days(p) = number of days since paper p's update_date
tau = recency time scale
```

The current daily feed uses `recency_score(...)`, which defaults to a 30-day
halflife-style decay. This design keeps the existing recency behavior.

### Centroid Coverage

```text
centroid_coverage(p | S, U) =
    1 / (1 + thread_count_S(nearest_thread(p)))
```

Where:

```text
thread_count_S(t) = number of selected papers in S whose nearest_thread is t
```

Examples:

```text
thread_count = 0 -> coverage = 1.000
thread_count = 1 -> coverage = 0.500
thread_count = 2 -> coverage = 0.333
thread_count = 3 -> coverage = 0.250
```

This rewards papers from under-covered user interests without forcing a hard
one-paper-per-centroid rule. The reward is strongest when a centroid has not yet
appeared in the feed, then decays smoothly.

### Cluster Saturation

```text
cluster_saturation(p | S) =
    cluster_count_S(cluster_id(p)) / max_per_cluster
```

Where:

```text
cluster_count_S(c) = number of selected papers in S from k-means cluster c
max_per_cluster = DAILY_MAX_PER_CLUSTER
```

The final selector should keep the existing hard cap:

```text
cluster_count_S(c) < DAILY_MAX_PER_CLUSTER
```

The saturation term is a soft penalty before the hard cap is reached. It makes a
candidate from an already-represented cluster slightly less attractive when
`delta` is high, but it does not ban it unless the hard cap is hit.

### No Selected-Paper Similarity Penalty

The formula intentionally excludes:

```text
max_{q in S} cosine(e_p, e_q)
```

That means papers are not penalized merely because they are similar to already
selected papers. Diversity is instead encouraged through:

1. more retrieval clusters as `delta` rises,
2. smoother coverage across user centroids,
3. cluster saturation and the existing hard per-cluster cap.

## Behavior By Delta

At `delta = 0`:

```text
score(p) = relevance(p, U) + alpha * recency(p)
```

The feed is focused. User-centroid coverage and cluster saturation do not affect
ranking.

At `delta = 1`:

```text
score(p | S, U) =
    relevance(p, U)
  + alpha * recency(p)
  + beta * centroid_coverage(p | S, U)
  - eta * cluster_saturation(p | S)
```

The feed still prioritizes relevance, but it more strongly favors under-covered
user centroids and slightly downweights clusters already represented in the feed.

For intermediate values, the effect is continuous:

```text
delta = 0.25 -> mild coverage pressure
delta = 0.50 -> moderate coverage pressure
delta = 0.75 -> strong coverage pressure
```

This removes the current hard threshold behavior around `delta > 0.5`.

## Greedy Selection Algorithm

The selector should be greedy because the score depends on the selected prefix
`S`.

Pseudo-code:

```text
selected = []
remaining = scored_candidates
thread_counts = Counter()
cluster_counts = Counter()

while len(selected) < n:
    best = None
    best_score = -infinity

    for candidate in remaining:
        if candidate.paper_id in seen_ids:
            continue
        if candidate.paper_id already selected:
            continue
        if cluster_counts[candidate.cluster_id] >= DAILY_MAX_PER_CLUSTER:
            continue

        base_score =
            candidate.raw_similarity
          + alpha * candidate.recency_score

        coverage =
            1 / (1 + thread_counts[candidate.nearest_thread])

        saturation =
            cluster_counts[candidate.cluster_id] / DAILY_MAX_PER_CLUSTER

        adjusted_score =
            base_score
          + delta * beta * coverage
          - delta * eta * saturation

        if adjusted_score > best_score:
            best = candidate
            best_score = adjusted_score

    if best is None:
        break

    selected.append(best)
    remove best from remaining
    thread_counts[best.nearest_thread] += 1
    cluster_counts[best.cluster_id] += 1
```

## Implementation Plan

### 1. Add Diversity Clamping

Add a small helper in the recommender layer:

```text
_clamp_diversity(diversity: float) -> float
```

Behavior:

```text
None or invalid values -> 0.5, if desired
finite values below 0.0 -> 0.0
finite values above 1.0 -> 1.0
valid values -> unchanged
```

Use the clamped value in:

```text
recommend(...)
select_with_relaxation(...)
find_nearest_clusters(...)
query_search.select_user_clusters(...)
```

The retrieval formula itself remains unchanged.

### 2. Preserve Retrieval Breadth

Keep the existing cluster-budget formula in `recommender/retrieve.py`:

```text
ceil(DAILY_CLUSTER_BUDGET_BASE
     + delta * DAILY_CLUSTER_BUDGET_DIVERSITY_MULTIPLIER)
```

No change to the intended behavior:

```text
delta = 0.0 -> 4 clusters
delta = 0.5 -> 8 clusters
delta = 1.0 -> 12 clusters
```

### 3. Replace Threshold-Based Early Coverage

In `recommender/engine.py`, replace the current special block:

```text
if diversity > 0.5 and k_u > 1:
    select early centroid coverage papers
```

with a single greedy scoring loop that always uses the same formula. At
`delta = 0`, the diversity terms become zero, so this still behaves like focused
ranking. At higher `delta`, coverage pressure increases smoothly.

### 4. Keep Cluster Hard Cap

Retain:

```text
DAILY_MAX_PER_CLUSTER = 2
```

and continue rejecting candidates once a selected cluster reaches that cap. The
new cluster saturation term only affects ranking before the cap is reached.

### 5. Add Score Metadata

For debugging and evaluation, enrich selected papers with:

```text
raw_similarity
recency_score
base_score
diversity_adjusted_score
centroid_coverage_bonus
cluster_saturation_penalty
nearest_centroid_id
```

This will make it easier to audit why a high-delta feed differs from a low-delta
feed.

### 6. Update Tests

Add or update tests for:

1. Diversity values below `0.0` and above `1.0` are clamped.
2. Retrieval cluster budget still follows the current formula.
3. At `delta = 0`, ordering matches relevance plus recency, subject to hard
   cluster caps.
4. At high `delta`, candidates from under-covered user centroids are promoted.
5. Similarity to already-selected papers is not used as a penalty.
6. The final feed never exceeds `DAILY_MAX_PER_CLUSTER`.

### 7. Evaluate The Change

Run offline sweeps across:

```text
delta in {0.0, 0.25, 0.5, 0.75, 1.0}
```

Track:

```text
NDCG@20
Precision@20
unique user centroids@20
thread coverage@20
unique clusters@20
cluster entropy@20
category entropy@20
mean raw similarity
```

Expected outcome:

```text
delta rises -> thread and cluster coverage rise
delta rises -> mean relevance declines mildly, not sharply
adjacent delta values -> feed changes smoothly
```

## Non-Goals

This design does not:

1. Change the offline pipeline.
2. Change k-means training or corpus embeddings.
3. Penalize selected-paper embedding similarity.
4. Replace the existing cluster retrieval formula.
5. Add personalized learning of `delta` from feedback.

