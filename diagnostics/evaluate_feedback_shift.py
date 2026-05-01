"""Visualize angle-to-liked feedback learning over synthetic trials.

This script follows the "Angle-to-Liked Trajectory -- Update-Rule Evaluation"
section of evaluation_plan_v1.pdf:

* for every like/save event, measure the nearest-centroid cosine before update
  as s_t = max_j u_t(j) dot e_p,
* convert it to theta_t = arccos(s_t),
* report the raw trajectory, a rolling mean with window w=10 events, and the
  last-quartile / first-quartile angle ratio.

It also logs lightweight top-5 feed metrics after each update so the geometric
shift can be compared with recommendation quality.
"""

from __future__ import annotations

import csv
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from recommender.engine import recommend
from user.profile import apply_feedback


DIM = 768
N_TRIALS = 10
N_STEPS = 12
SERVE_K = 5
ROLLING_WINDOW = 10
BASE_SEED = 20260501
OUT_DIR = ROOT / "diagnostics" / "feedback_shift_outputs"


@dataclass
class SyntheticIndex:
    embeddings: np.ndarray
    cluster_ids: np.ndarray
    centroids: np.ndarray
    paper_meta: list[dict]


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return (v / max(float(np.linalg.norm(v)), eps)).astype(np.float32)


def unit_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, eps)).astype(np.float32)


def orthogonal_direction(
    rng: np.random.Generator,
    bases: list[np.ndarray],
) -> np.ndarray:
    v = rng.standard_normal(DIM).astype(np.float32)
    for base in bases:
        v = v - float(v @ base) * base
    return unit(v)


def around_topic(
    rng: np.random.Generator,
    topic: np.ndarray,
    n: int,
    noise: float,
) -> np.ndarray:
    rows = []
    for _ in range(n):
        eps = rng.standard_normal(DIM).astype(np.float32)
        eps = eps - float(eps @ topic) * topic
        rows.append(unit(topic + noise * unit(eps)))
    return np.stack(rows).astype(np.float32)


def build_trial(trial_id: int) -> tuple[SyntheticIndex, dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(BASE_SEED + 997 * trial_id)

    source = orthogonal_direction(rng, [])
    target_axis = orthogonal_direction(rng, [source])
    target_source_cos = rng.uniform(0.30, 0.55)
    target = unit(
        target_source_cos * source
        + math.sqrt(1.0 - target_source_cos**2) * target_axis
    )
    stable = orthogonal_direction(rng, [source, target])
    distractor = orthogonal_direction(rng, [source, target, stable])

    topic_specs = [
        ("clicked_target", target, 120, range(0, 8), 0.22),
        ("old_source", source, 90, range(8, 14), 0.24),
        ("stable_thread", stable, 90, range(14, 20), 0.22),
        ("distractor", distractor, 90, range(20, 26), 0.30),
    ]

    embeddings: list[np.ndarray] = []
    cluster_ids: list[int] = []
    paper_meta: list[dict] = []
    for label, topic, n_papers, clusters, noise in topic_specs:
        rows = around_topic(rng, topic, n_papers, noise)
        cluster_cycle = list(clusters)
        for i, emb in enumerate(rows):
            cid = cluster_cycle[i % len(cluster_cycle)]
            paper_id = f"trial{trial_id:02d}-{label}-{i:03d}"
            embeddings.append(emb)
            cluster_ids.append(cid)
            paper_meta.append(
                {
                    "id": paper_id,
                    "arxiv_id": paper_id,
                    "title": f"{label.replace('_', ' ').title()} {i}",
                    "abstract": f"Synthetic evaluation paper for {label}.",
                    "categories": label,
                    "update_date": "2026-04-30T00:00:00",
                    "cluster_id": cid,
                    "synthetic_label": label,
                }
            )

    embeddings_arr = np.stack(embeddings).astype(np.float32)
    cluster_ids_arr = np.asarray(cluster_ids, dtype=np.int32)
    centroids = []
    for cid in sorted(set(cluster_ids)):
        centroids.append(unit(embeddings_arr[cluster_ids_arr == cid].mean(axis=0)))

    initial_user_centroids = unit_rows(
        np.stack(
            [
                unit(0.90 * source + 0.10 * target),
                stable,
            ]
        )
    ).astype(np.float32)

    index = SyntheticIndex(
        embeddings=embeddings_arr,
        cluster_ids=cluster_ids_arr,
        centroids=np.stack(centroids).astype(np.float32),
        paper_meta=paper_meta,
    )
    topics = {
        "source": source,
        "target": target,
        "stable": stable,
        "distractor": distractor,
    }
    return index, topics, initial_user_centroids


def relevance_grade(label: str) -> int:
    """Synthetic three-point relevance grade, matching the PDF's NDCG@5 setup."""
    if label in {"clicked_target", "stable_thread"}:
        return 2
    if label == "old_source":
        return 1
    return 0


def ndcg_at_k(grades: list[int], k: int = SERVE_K) -> float:
    used = grades[:k]
    dcg = sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(used))
    ideal = sorted(used, reverse=True)
    idcg = sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(ideal))
    return 0.0 if idcg <= 0 else float(dcg / idcg)


def feed_metrics(feed: list[dict], index: SyntheticIndex) -> dict[str, float]:
    grades = [relevance_grade(p["synthetic_label"]) for p in feed[:SERVE_K]]
    labels = [p["synthetic_label"] for p in feed[:SERVE_K]]
    by_id = {meta["id"]: i for i, meta in enumerate(index.paper_meta)}
    embs = np.stack([index.embeddings[by_id[p["id"]]] for p in feed[:SERVE_K]])
    if len(embs) > 1:
        sims = embs @ embs.T
        redundancy = float(sims[np.triu_indices(len(embs), k=1)].mean())
    else:
        redundancy = 0.0
    counts = Counter(labels)
    return {
        "precision_at_5": float(np.mean([grade >= 1 for grade in grades])),
        "strong_precision_at_5": float(np.mean([grade == 2 for grade in grades])),
        "mean_relevance_at_5": float(np.mean([grade / 2 for grade in grades])),
        "ndcg_at_5": ndcg_at_k(grades),
        "target_share_at_5": counts["clicked_target"] / SERVE_K,
        "redundancy_at_5": redundancy,
    }


def rolling_mean(values: list[float], window: int) -> list[float | None]:
    out: list[float | None] = []
    for i in range(len(values)):
        if i + 1 < window:
            out.append(None)
        else:
            out.append(float(np.mean(values[i + 1 - window : i + 1])))
    return out


def run_trials() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    summaries: list[dict] = []

    for trial_id in range(1, N_TRIALS + 1):
        rng = np.random.default_rng(BASE_SEED + 131 * trial_id)
        index, topics, centroids = build_trial(trial_id)
        initial_centroids = centroids.copy()

        target_indices = [
            i
            for i, meta in enumerate(index.paper_meta)
            if meta["synthetic_label"] == "clicked_target"
        ]
        rng.shuffle(target_indices)
        target_indices = target_indices[:N_STEPS]
        signals = rng.choice(["like", "save"], size=N_STEPS, p=[0.55, 0.45]).tolist()

        clicked_ids: set[str] = set()
        theta_values: list[float] = []
        cosine_values: list[float] = []
        target_pool = np.stack(
            [
                index.embeddings[i]
                for i, meta in enumerate(index.paper_meta)
                if meta["synthetic_label"] == "clicked_target"
            ]
        )

        for step, (paper_idx, signal) in enumerate(zip(target_indices, signals), start=1):
            paper_emb = index.embeddings[paper_idx]
            pre_sims = centroids @ paper_emb
            s_pre = float(pre_sims.max())
            nearest_pre = int(pre_sims.argmax())
            theta_pre_deg = float(np.degrees(np.arccos(np.clip(s_pre, -1.0, 1.0))))

            clicked_ids.add(index.paper_meta[paper_idx]["id"])
            centroids = apply_feedback(centroids, paper_emb, signal)

            s_post = float((centroids @ paper_emb).max())
            theta_post_deg = float(np.degrees(np.arccos(np.clip(s_post, -1.0, 1.0))))
            future_liked_similarity = float((target_pool @ centroids.T).max(axis=1).mean())
            off_thread_drift = float(1.0 - centroids[1] @ initial_centroids[1])

            feed = recommend(
                centroids,
                seen_ids=clicked_ids,
                index=index,  # type: ignore[arg-type]
                diversity=0.7,
                n=SERVE_K,
            )
            metrics = feed_metrics(feed, index)

            theta_values.append(theta_pre_deg)
            cosine_values.append(s_pre)
            rows.append(
                {
                    "trial": trial_id,
                    "step": step,
                    "signal": signal,
                    "clicked_paper_id": index.paper_meta[paper_idx]["id"],
                    "nearest_centroid_pre": nearest_pre,
                    "pre_update_max_cosine_st": s_pre,
                    "pre_update_theta_deg": theta_pre_deg,
                    "post_update_max_cosine_same_paper": s_post,
                    "post_update_theta_same_paper_deg": theta_post_deg,
                    "future_liked_mean_similarity": future_liked_similarity,
                    "off_thread_drift_cosine_distance": off_thread_drift,
                    **metrics,
                }
            )

        quartile = max(1, N_STEPS // 4)
        first_q = float(np.mean(theta_values[:quartile]))
        last_q = float(np.mean(theta_values[-quartile:]))
        summaries.append(
            {
                "trial": trial_id,
                "first_quartile_theta_mean": first_q,
                "last_quartile_theta_mean": last_q,
                "angle_ratio_last_over_first": last_q / first_q,
                "cosine_gain_last_over_first": float(
                    np.mean(cosine_values[-quartile:]) - np.mean(cosine_values[:quartile])
                ),
            }
        )

    return rows, summaries


def mean_by_step(rows: list[dict], key: str) -> list[float]:
    values = []
    for step in range(1, N_STEPS + 1):
        vals = [float(row[key]) for row in rows if row["step"] == step]
        values.append(float(np.mean(vals)))
    return values


def write_csv(rows: list[dict], summaries: list[dict]) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = OUT_DIR / "angle_to_liked_trials.csv"
    summary_path = OUT_DIR / "angle_to_liked_summary.csv"

    with data_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    return data_path, summary_path


def build_chart(rows: list[dict], summaries: list[dict]) -> Path:
    steps = list(range(1, N_STEPS + 1))
    colors = [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#72B7B2",
        "#B279A2",
        "#FF9DA6",
        "#9D755D",
        "#BAB0AC",
        "#2F4B7C",
    ]

    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=(
            "Pre-update angle to liked paper, theta_t",
            "Pre-update nearest-centroid cosine, s_t",
            "Mean Relevance@5 after update",
            "Precision@5 and NDCG@5 after update",
            "Future liked-paper mean similarity",
            "Target share and redundancy in top-5",
            "Raw theta_t scatter + rolling mean (w=10 events)",
            "Angle ratio: last quartile / first quartile",
        ),
        vertical_spacing=0.09,
        horizontal_spacing=0.10,
    )

    for trial_id in range(1, N_TRIALS + 1):
        trial_rows = [row for row in rows if row["trial"] == trial_id]
        color = colors[(trial_id - 1) % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=[row["pre_update_theta_deg"] for row in trial_rows],
                mode="lines+markers",
                line={"width": 1, "color": color},
                marker={"size": 4},
                opacity=0.35,
                name=f"trial {trial_id} theta",
                legendgroup=f"trial{trial_id}",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=[row["pre_update_max_cosine_st"] for row in trial_rows],
                mode="lines",
                line={"width": 1, "color": color},
                opacity=0.25,
                name=f"trial {trial_id} cosine",
                legendgroup=f"trial{trial_id}",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    mean_theta = mean_by_step(rows, "pre_update_theta_deg")
    mean_cosine = mean_by_step(rows, "pre_update_max_cosine_st")
    mean_relevance = mean_by_step(rows, "mean_relevance_at_5")
    mean_precision = mean_by_step(rows, "precision_at_5")
    mean_strong_precision = mean_by_step(rows, "strong_precision_at_5")
    mean_ndcg = mean_by_step(rows, "ndcg_at_5")
    mean_future_similarity = mean_by_step(rows, "future_liked_mean_similarity")
    mean_target_share = mean_by_step(rows, "target_share_at_5")
    mean_redundancy = mean_by_step(rows, "redundancy_at_5")

    mean_style = {"width": 4, "color": "#111111"}
    fig.add_trace(
        go.Scatter(x=steps, y=mean_theta, mode="lines+markers", name="mean theta", line=mean_style),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=steps, y=mean_cosine, mode="lines+markers", name="mean s_t", line=mean_style),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean_relevance,
            mode="lines+markers",
            name="mean relevance@5",
            line={"width": 4, "color": "#2F855A"},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean_precision,
            mode="lines+markers",
            name="precision@5",
            line={"width": 3, "color": "#4C78A8"},
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean_strong_precision,
            mode="lines+markers",
            name="strong precision@5",
            line={"width": 3, "color": "#F58518"},
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean_ndcg,
            mode="lines+markers",
            name="NDCG@5",
            line={"width": 3, "color": "#B279A2"},
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean_future_similarity,
            mode="lines+markers",
            name="future liked similarity",
            line={"width": 4, "color": "#2B6CB0"},
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean_target_share,
            mode="lines+markers",
            name="target share@5",
            line={"width": 3, "color": "#38A169"},
        ),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean_redundancy,
            mode="lines+markers",
            name="redundancy@5",
            line={"width": 3, "color": "#E45756"},
        ),
        row=3,
        col=2,
    )

    flattened = sorted(rows, key=lambda row: (row["trial"], row["step"]))
    event_x = list(range(1, len(flattened) + 1))
    theta_events = [float(row["pre_update_theta_deg"]) for row in flattened]
    theta_roll = rolling_mean(theta_events, ROLLING_WINDOW)
    fig.add_trace(
        go.Scatter(
            x=event_x,
            y=theta_events,
            mode="markers",
            name="raw theta events",
            marker={"size": 5, "color": "#4C78A8", "opacity": 0.45},
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=event_x,
            y=theta_roll,
            mode="lines",
            name="rolling mean theta, w=10",
            line={"width": 4, "color": "#111111"},
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[f"T{summary['trial']}" for summary in summaries],
            y=[summary["angle_ratio_last_over_first"] for summary in summaries],
            name="last/first theta ratio",
            marker_color="#4C78A8",
        ),
        row=4,
        col=2,
    )
    mean_ratio = float(np.mean([summary["angle_ratio_last_over_first"] for summary in summaries]))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#999999", row=4, col=2)
    fig.add_hline(
        y=mean_ratio,
        line_dash="dot",
        line_color="#111111",
        annotation_text=f"mean ratio={mean_ratio:.3f}",
        row=4,
        col=2,
    )

    fig.update_xaxes(title_text="interaction step", row=1, col=1)
    fig.update_xaxes(title_text="interaction step", row=1, col=2)
    fig.update_xaxes(title_text="interaction step", row=2, col=1)
    fig.update_xaxes(title_text="interaction step", row=2, col=2)
    fig.update_xaxes(title_text="interaction step", row=3, col=1)
    fig.update_xaxes(title_text="interaction step", row=3, col=2)
    fig.update_xaxes(title_text="positive feedback event", row=4, col=1)
    fig.update_xaxes(title_text="trial", row=4, col=2)

    fig.update_yaxes(title_text="degrees, lower is better", row=1, col=1)
    fig.update_yaxes(title_text="cosine, higher is better", row=1, col=2)
    fig.update_yaxes(title_text="0-1", row=2, col=1, range=[0, 1.02])
    fig.update_yaxes(title_text="0-1", row=2, col=2, range=[0, 1.02])
    fig.update_yaxes(title_text="cosine", row=3, col=1, range=[0, 1.02])
    fig.update_yaxes(title_text="0-1 / cosine", row=3, col=2)
    fig.update_yaxes(title_text="degrees", row=4, col=1)
    fig.update_yaxes(title_text="ratio, lower is better", row=4, col=2)

    fig.update_layout(
        title=(
            "Angle-to-Liked Trajectory Across 10 Synthetic Trials "
            "(12 like/save updates each)"
        ),
        width=1500,
        height=1500,
        template="plotly_white",
        legend={"orientation": "h", "y": -0.05},
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = OUT_DIR / "angle_to_liked_trajectory.html"
    fig.write_html(html_path, include_plotlyjs="cdn")

    png_path = OUT_DIR / "angle_to_liked_trajectory.png"
    try:
        temp_dir = OUT_DIR / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TMP"] = str(temp_dir)
        os.environ["TEMP"] = str(temp_dir)
        os.environ["TMPDIR"] = str(temp_dir)
        fig.write_image(png_path, scale=2)
    except Exception as exc:  # pragma: no cover - depends on local kaleido setup.
        print(f"PNG export skipped: {exc}")

    return html_path


def build_static_png(rows: list[dict], summaries: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 1800, 1350
    margin = 70
    gap_x = 80
    gap_y = 95
    panel_w = (width - 2 * margin - gap_x) // 2
    panel_h = (height - 2 * margin - 2 * gap_y) // 3
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    def panel_bounds(row: int, col: int) -> tuple[int, int, int, int]:
        x0 = margin + col * (panel_w + gap_x)
        y0 = margin + row * (panel_h + gap_y)
        return x0, y0, x0 + panel_w, y0 + panel_h

    def draw_panel(
        row: int,
        col: int,
        title: str,
        series: list[tuple[str, list[float], str]],
        y_min: float | None = None,
        y_max: float | None = None,
    ) -> None:
        x0, y0, x1, y1 = panel_bounds(row, col)
        plot_x0, plot_y0 = x0 + 58, y0 + 34
        plot_x1, plot_y1 = x1 - 20, y1 - 42
        draw.text((x0, y0), title, fill="#111111", font=font)
        all_values = [v for _name, vals, _color in series for v in vals]
        lo = min(all_values) if y_min is None else y_min
        hi = max(all_values) if y_max is None else y_max
        if abs(hi - lo) < 1e-9:
            hi = lo + 1.0
        draw.rectangle((plot_x0, plot_y0, plot_x1, plot_y1), outline="#D0D0D0")
        for tick in range(5):
            y = plot_y1 - tick * (plot_y1 - plot_y0) / 4
            value = lo + tick * (hi - lo) / 4
            draw.line((plot_x0, y, plot_x1, y), fill="#EEEEEE")
            draw.text((x0, y - 6), f"{value:.2f}", fill="#555555", font=font)
        for step in (1, 4, 8, 12):
            x = plot_x0 + (step - 1) * (plot_x1 - plot_x0) / (N_STEPS - 1)
            draw.text((x - 8, plot_y1 + 10), str(step), fill="#555555", font=font)
        for idx, (name, vals, color) in enumerate(series):
            points = []
            for i, val in enumerate(vals):
                x = plot_x0 + i * (plot_x1 - plot_x0) / (len(vals) - 1)
                y = plot_y1 - (val - lo) * (plot_y1 - plot_y0) / (hi - lo)
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=color, width=4)
            for x, y in points:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
            draw.text((plot_x0 + idx * 185, y1 - 24), name, fill=color, font=font)

    steps = list(range(1, N_STEPS + 1))
    del steps
    draw_panel(
        0,
        0,
        "Pre-update theta_t (degrees, lower is better)",
        [("mean theta", mean_by_step(rows, "pre_update_theta_deg"), "#111111")],
    )
    draw_panel(
        0,
        1,
        "Pre-update s_t cosine (higher is better)",
        [("mean cosine", mean_by_step(rows, "pre_update_max_cosine_st"), "#2B6CB0")],
        y_min=0,
        y_max=1,
    )
    draw_panel(
        1,
        0,
        "Mean Relevance@5 after each update",
        [("mean rel", mean_by_step(rows, "mean_relevance_at_5"), "#2F855A")],
        y_min=0,
        y_max=1,
    )
    draw_panel(
        1,
        1,
        "Precision/NDCG@5 after each update",
        [
            ("strong P@5", mean_by_step(rows, "strong_precision_at_5"), "#F58518"),
            ("NDCG@5", mean_by_step(rows, "ndcg_at_5"), "#7B61A8"),
        ],
        y_min=0,
        y_max=1,
    )
    draw_panel(
        2,
        0,
        "Future liked-paper mean similarity",
        [("future sim", mean_by_step(rows, "future_liked_mean_similarity"), "#4C78A8")],
        y_min=0,
        y_max=1,
    )

    x0, y0, x1, y1 = panel_bounds(2, 1)
    draw.text((x0, y0), "Angle ratio by trial: last quartile / first quartile", fill="#111111", font=font)
    plot_x0, plot_y0 = x0 + 58, y0 + 34
    plot_x1, plot_y1 = x1 - 20, y1 - 42
    draw.rectangle((plot_x0, plot_y0, plot_x1, plot_y1), outline="#D0D0D0")
    ratios = [summary["angle_ratio_last_over_first"] for summary in summaries]
    y_max = max(1.0, max(ratios) * 1.15)
    zero_y = plot_y1
    for idx, ratio in enumerate(ratios):
        bar_w = (plot_x1 - plot_x0) / (len(ratios) * 1.7)
        x = plot_x0 + idx * (plot_x1 - plot_x0) / len(ratios) + bar_w * 0.35
        bar_top = plot_y1 - ratio * (plot_y1 - plot_y0) / y_max
        draw.rectangle((x, bar_top, x + bar_w, zero_y), fill="#4C78A8")
        draw.text((x - 2, plot_y1 + 10), f"T{idx+1}", fill="#555555", font=font)
    y_one = plot_y1 - (plot_y1 - plot_y0) / y_max
    draw.line((plot_x0, y_one, plot_x1, y_one), fill="#999999", width=2)
    draw.text((plot_x0, y_one - 16), "1.0", fill="#555555", font=font)
    mean_ratio = float(np.mean(ratios))
    y_mean = plot_y1 - mean_ratio * (plot_y1 - plot_y0) / y_max
    draw.line((plot_x0, y_mean, plot_x1, y_mean), fill="#111111", width=3)
    draw.text((plot_x0 + 8, y_mean + 6), f"mean={mean_ratio:.3f}", fill="#111111", font=font)

    title = "Angle-to-Liked Trajectory, 10 Trials x 12 Like/Save Updates"
    draw.text((margin, 24), title, fill="#111111", font=font)
    png_path = OUT_DIR / "angle_to_liked_trajectory.png"
    image.save(png_path)
    return png_path


def main() -> None:
    rows, summaries = run_trials()
    data_path, summary_path = write_csv(rows, summaries)
    chart_path = build_chart(rows, summaries)
    png_path = build_static_png(rows, summaries)

    mean_first = float(np.mean([s["first_quartile_theta_mean"] for s in summaries]))
    mean_last = float(np.mean([s["last_quartile_theta_mean"] for s in summaries]))
    mean_ratio = float(np.mean([s["angle_ratio_last_over_first"] for s in summaries]))
    mean_cosine_gain = float(np.mean([s["cosine_gain_last_over_first"] for s in summaries]))
    final_mean_relevance = float(
        np.mean([row["mean_relevance_at_5"] for row in rows if row["step"] == N_STEPS])
    )
    final_ndcg = float(np.mean([row["ndcg_at_5"] for row in rows if row["step"] == N_STEPS]))

    print("Angle-to-liked update-rule evaluation")
    print(f"trials: {N_TRIALS}")
    print(f"steps_per_trial: {N_STEPS}")
    print(f"rows: {len(rows)}")
    print(f"mean_first_quartile_theta_deg: {mean_first:.3f}")
    print(f"mean_last_quartile_theta_deg: {mean_last:.3f}")
    print(f"mean_angle_ratio_last_over_first: {mean_ratio:.3f}")
    print(f"mean_cosine_gain_last_over_first: {mean_cosine_gain:.3f}")
    print(f"final_mean_relevance_at_5: {final_mean_relevance:.3f}")
    print(f"final_ndcg_at_5: {final_ndcg:.3f}")
    print(f"data_csv: {data_path}")
    print(f"summary_csv: {summary_path}")
    print(f"chart_html: {chart_path}")
    print(f"chart_png: {png_path}")


if __name__ == "__main__":
    main()
