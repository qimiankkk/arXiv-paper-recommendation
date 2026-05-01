"""Generate Angle-to-Liked trial data from real arXiv embedding artifacts.

Unlike the earlier random synthetic diagnostic, this script samples real paper
vectors from data/embeddings.npy and real k-means cluster IDs from
data/cluster_ids.npy. It then replays 10 trials x 12 like/save events through
the production EMA update rule and recommendation code.

The output CSV keeps the same columns as the uploaded step data so it can be
evaluated with the same Angle-to-Liked plotting script.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from recommender.engine import recommend
from user.profile import apply_feedback
import diagnostics.plot_uploaded_angle_to_liked as plot_angle


N_TRIALS = 10
N_STEPS = 12
FEED_K = 20
BASE_SEED = 20260501
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "diagnostics" / "realistic_angle_to_liked_outputs"
DATA_CSV = OUT_DIR / "angle_to_liked_trials_step_data.csv"


@dataclass
class LocalPaperIndex:
    embeddings: np.ndarray
    cluster_ids: np.ndarray
    centroids: np.ndarray
    paper_meta: list[dict]


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return (v / max(float(np.linalg.norm(v)), eps)).astype(np.float32)


def build_cluster_members(cluster_ids: np.ndarray) -> dict[int, np.ndarray]:
    members: dict[int, np.ndarray] = {}
    for cid in np.unique(cluster_ids):
        indices = np.where(cluster_ids == cid)[0]
        if len(indices) > 0:
            members[int(cid)] = indices
    return members


def sample_from_cluster(
    rng: np.random.Generator,
    members: dict[int, np.ndarray],
    cluster_id: int,
    count: int,
) -> np.ndarray:
    indices = members[int(cluster_id)]
    replace = len(indices) < count
    return rng.choice(indices, size=count, replace=replace)


def select_trial_clusters(
    trial_id: int,
    eligible_clusters: list[int],
    cluster_centroids: np.ndarray,
) -> tuple[int, int, int, list[int]]:
    target_cluster = eligible_clusters[(trial_id * 37) % len(eligible_clusters)]
    target_centroid = unit(cluster_centroids[target_cluster])
    sims = cluster_centroids @ target_centroid
    ranked = [int(cid) for cid in np.argsort(sims)[::-1] if int(cid) != target_cluster]
    source_cluster = ranked[4 + (trial_id % 4)]
    stable_cluster = ranked[-(8 + trial_id)]
    distractor_clusters = ranked[40 + trial_id : 50 + trial_id]
    return target_cluster, source_cluster, stable_cluster, distractor_clusters


def make_local_index(
    rng: np.random.Generator,
    embeddings: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_centroids: np.ndarray,
    members: dict[int, np.ndarray],
    target_cluster: int,
    source_cluster: int,
    stable_cluster: int,
    distractor_clusters: list[int],
) -> tuple[LocalPaperIndex, np.ndarray, np.ndarray]:
    target_centroid = unit(cluster_centroids[target_cluster])

    target_pool = sample_from_cluster(rng, members, target_cluster, 220)
    source_pool = sample_from_cluster(rng, members, source_cluster, 160)
    stable_pool = sample_from_cluster(rng, members, stable_cluster, 160)
    distractor_pools = [
        sample_from_cluster(rng, members, cid, 50) for cid in distractor_clusters
    ]

    all_indices = np.unique(
        np.concatenate([target_pool, source_pool, stable_pool, *distractor_pools])
    )
    local_embeddings = np.asarray(embeddings[all_indices], dtype=np.float32)
    local_cluster_ids = np.asarray(cluster_ids[all_indices], dtype=np.int32)
    paper_meta = [
        {
            "id": f"real-{int(real_idx)}",
            "arxiv_id": f"real-{int(real_idx)}",
            "title": f"Real arXiv embedding paper {int(real_idx)}",
            "abstract": "Real arXiv embedding sampled for Angle-to-Liked replay.",
            "categories": "sampled",
            "update_date": "2026-04-30T00:00:00",
            "cluster_id": int(cid),
            "real_embedding_index": int(real_idx),
        }
        for real_idx, cid in zip(all_indices, local_cluster_ids)
    ]
    index = LocalPaperIndex(
        embeddings=local_embeddings,
        cluster_ids=local_cluster_ids,
        centroids=cluster_centroids.astype(np.float32),
        paper_meta=paper_meta,
    )

    local_by_real = {int(real_idx): i for i, real_idx in enumerate(all_indices)}
    target_local = np.array(
        [local_by_real[int(real_idx)] for real_idx in target_pool if int(real_idx) in local_by_real],
        dtype=int,
    )
    # Prefer liked papers that are realistic members of the target area, not
    # necessarily only the mathematically nearest points.
    target_sims = local_embeddings[target_local] @ target_centroid
    middle = np.argsort(target_sims)[::-1][20:120]
    liked_local_candidates = target_local[middle]

    source_centroid = unit(cluster_centroids[source_cluster])
    stable_centroid = unit(cluster_centroids[stable_cluster])
    initial_centroids = np.stack(
        [
            unit(0.68 * source_centroid + 0.32 * target_centroid),
            stable_centroid,
        ]
    ).astype(np.float32)

    return index, target_centroid, initial_centroids, liked_local_candidates


def relevance_grade(
    meta: dict,
    emb: np.ndarray,
    target_centroid: np.ndarray,
    target_cluster: int,
    soft_threshold: float,
) -> float:
    if int(meta["cluster_id"]) == int(target_cluster):
        return 2.0
    if float(emb @ target_centroid) >= soft_threshold:
        return 1.0
    return 0.0


def feed_metrics(
    feed: list[dict],
    centroids: np.ndarray,
    index: LocalPaperIndex,
    target_centroid: np.ndarray,
    target_cluster: int,
    soft_threshold: float,
) -> dict[str, float]:
    by_id = {meta["id"]: i for i, meta in enumerate(index.paper_meta)}
    feed_indices = [by_id[item["id"]] for item in feed[:FEED_K]]
    if not feed_indices:
        return {
            "feed_mean_similarity": 0.0,
            "feed_mean_similarity_to_target": 0.0,
            "feed_mean_relevance_at_20": 0.0,
            "feed_strict_precision_at_20": 0.0,
            "feed_soft_precision_at_20": 0.0,
        }

    feed_embeddings = index.embeddings[feed_indices]
    max_sims = (feed_embeddings @ centroids.T).max(axis=1)
    target_sims = feed_embeddings @ target_centroid
    grades = np.array(
        [
            relevance_grade(index.paper_meta[i], index.embeddings[i], target_centroid, target_cluster, soft_threshold)
            for i in feed_indices
        ],
        dtype=np.float32,
    )
    return {
        "feed_mean_similarity": float(np.mean(max_sims)),
        "feed_mean_similarity_to_target": float(np.mean(target_sims)),
        "feed_mean_relevance_at_20": float(np.mean(grades / 2.0)),
        "feed_strict_precision_at_20": float(np.mean(grades >= 1.5)),
        "feed_soft_precision_at_20": float(np.mean(grades >= 0.8)),
    }


def generate_rows() -> list[dict]:
    embeddings = np.load(DATA_DIR / "embeddings.npy", mmap_mode="r")
    cluster_ids = np.load(DATA_DIR / "cluster_ids.npy")
    cluster_centroids = np.load(DATA_DIR / "centroids.npy").astype(np.float32)
    cluster_centroids = np.stack([unit(row) for row in cluster_centroids]).astype(np.float32)
    members = build_cluster_members(cluster_ids)
    eligible_clusters = sorted(
        cid for cid, indices in members.items() if len(indices) >= 260
    )
    if len(eligible_clusters) < N_TRIALS:
        raise RuntimeError("Not enough populated clusters for realistic trials.")

    rows: list[dict] = []
    for trial in range(1, N_TRIALS + 1):
        rng = np.random.default_rng(BASE_SEED + 211 * trial)
        target_cluster, source_cluster, stable_cluster, distractor_clusters = (
            select_trial_clusters(trial, eligible_clusters, cluster_centroids)
        )
        index, target_centroid, centroids, liked_candidates = make_local_index(
            rng,
            embeddings,
            cluster_ids,
            cluster_centroids,
            members,
            target_cluster,
            source_cluster,
            stable_cluster,
            distractor_clusters,
        )
        liked_candidates = rng.choice(
            liked_candidates,
            size=N_STEPS,
            replace=len(liked_candidates) < N_STEPS,
        )
        feedback_types = rng.choice(["like", "save"], size=N_STEPS, p=[0.55, 0.45])
        clicked_ids: set[str] = set()
        non_target_sims = index.embeddings[index.cluster_ids != target_cluster] @ target_centroid
        soft_threshold = float(np.quantile(non_target_sims, 0.94))

        for step, (paper_idx, feedback_type) in enumerate(
            zip(liked_candidates, feedback_types),
            start=1,
        ):
            paper_embedding = index.embeddings[int(paper_idx)]
            pre_sims = centroids @ paper_embedding
            pre_cosine = float(pre_sims.max())
            nearest_centroid = int(pre_sims.argmax())
            pre_angle = float(
                math.degrees(math.acos(max(-1.0, min(1.0, pre_cosine))))
            )

            picked_grade = relevance_grade(
                index.paper_meta[int(paper_idx)],
                paper_embedding,
                target_centroid,
                target_cluster,
                soft_threshold,
            )
            picked_target_similarity = float(paper_embedding @ target_centroid)

            clicked_ids.add(index.paper_meta[int(paper_idx)]["id"])
            centroids = apply_feedback(centroids, paper_embedding, str(feedback_type))
            feed = recommend(
                centroids,
                seen_ids=clicked_ids,
                index=index,  # type: ignore[arg-type]
                diversity=0.7,
                n=FEED_K,
            )
            metrics = feed_metrics(
                feed,
                centroids,
                index,
                target_centroid,
                target_cluster,
                soft_threshold,
            )

            rows.append(
                {
                    "trial": trial,
                    "step": step,
                    "feedback_type": str(feedback_type),
                    "preupdate_max_cosine_to_liked": pre_cosine,
                    "preupdate_angle_deg_to_liked": pre_angle,
                    "nearest_centroid_idx": nearest_centroid,
                    **metrics,
                    "picked_paper_true_relevance": picked_grade / 2.0,
                    "picked_paper_target_similarity": picked_target_similarity,
                }
            )

    return rows


def write_step_data(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with DATA_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = generate_rows()
    write_step_data(rows)

    # Reuse the validated Angle-to-Liked summarization and plotting functions
    # with this realistic output directory.
    plot_angle.OUT_DIR = OUT_DIR
    plot_angle.OUTPUT_PREFIX = "realistic_angle_to_liked"
    plot_angle.CHART_TITLE = (
        "Real arXiv-Embedding Angle-to-Liked Update-Rule Evaluation "
        "(10 trials x 12 like/save events)"
    )
    plot_angle.STATIC_TITLE = (
        "Real arXiv-Embedding Angle-to-Liked Evaluation, "
        "10 Trials x 12 Like/Save Events"
    )
    summaries, aggregate = plot_angle.summarize(rows)
    summary_csv, aggregate_csv = plot_angle.write_summary(summaries, aggregate)
    html_path = plot_angle.build_interactive_chart(rows, summaries, aggregate)
    png_path = plot_angle.build_static_png(rows, summaries, aggregate)

    print("Realistic arXiv-embedding Angle-to-Liked evaluation")
    print(f"rows: {len(rows)}")
    print(f"trials: {N_TRIALS}")
    print(f"steps_per_trial: {N_STEPS}")
    for key, value in aggregate.items():
        print(f"{key}: {value:.6f}")
    print(f"data_csv: {DATA_CSV}")
    print(f"summary_csv: {summary_csv}")
    print(f"aggregate_csv: {aggregate_csv}")
    print(f"chart_html: {html_path}")
    print(f"chart_png: {png_path}")


if __name__ == "__main__":
    main()
