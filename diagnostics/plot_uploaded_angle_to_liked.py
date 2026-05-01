"""Plot uploaded Angle-to-Liked trajectory step data.

The script validates that the uploaded CSV matches the evaluation protocol from
evaluation_plan_v1.pdf before plotting:

* 10 trials
* 12 positive feedback steps per trial
* pre-update nearest-centroid cosine s_t
* pre-update angle theta_t = arccos(s_t)

If the CSV does not look like centroid/embedding movement data for the
Angle-to-Liked evaluation, the script aborts.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = Path(r"C:\Users\emily\Downloads\angle_to_liked_trials_step_data.csv")
OUT_DIR = ROOT / "diagnostics" / "uploaded_angle_to_liked_outputs"
OUTPUT_PREFIX = "uploaded_angle_to_liked"
CHART_TITLE = (
    "Uploaded Angle-to-Liked Update-Rule Evaluation "
    "(10 trials x 12 like/save events)"
)
STATIC_TITLE = "Uploaded Angle-to-Liked Evaluation, 10 Trials x 12 Like/Save Events"
N_TRIALS = 10
N_STEPS = 12
ROLLING_WINDOW = 10

REQUIRED_COLUMNS = {
    "trial",
    "step",
    "feedback_type",
    "preupdate_max_cosine_to_liked",
    "preupdate_angle_deg_to_liked",
    "nearest_centroid_idx",
    "feed_mean_similarity",
    "feed_mean_similarity_to_target",
    "feed_mean_relevance_at_20",
    "feed_strict_precision_at_20",
    "feed_soft_precision_at_20",
    "picked_paper_true_relevance",
    "picked_paper_target_similarity",
}


def abort(message: str) -> None:
    raise SystemExit(f"ABORTED: {message}")


def load_and_validate(path: Path) -> list[dict]:
    if not path.exists():
        abort(f"CSV not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            abort(f"CSV is missing required columns: {sorted(missing)}")

        rows = []
        for raw in reader:
            row: dict = {"feedback_type": raw["feedback_type"]}
            for key, value in raw.items():
                if key == "feedback_type":
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    abort(f"Column {key!r} contains non-numeric value {value!r}")
            rows.append(row)

    if len(rows) != N_TRIALS * N_STEPS:
        abort(f"expected {N_TRIALS * N_STEPS} rows, found {len(rows)}")

    trials = sorted({int(row["trial"]) for row in rows})
    if trials != list(range(1, N_TRIALS + 1)):
        abort(f"expected trials 1..{N_TRIALS}, found {trials}")

    for trial in trials:
        trial_steps = sorted(
            int(row["step"]) for row in rows if int(row["trial"]) == trial
        )
        if trial_steps != list(range(1, N_STEPS + 1)):
            abort(f"trial {trial} does not have steps 1..{N_STEPS}")

    bad_feedback = sorted(
        {row["feedback_type"] for row in rows}
        - {"like", "save"}
    )
    if bad_feedback:
        abort(f"Angle-to-Liked should use only like/save events, found {bad_feedback}")

    for row in rows:
        cosine = row["preupdate_max_cosine_to_liked"]
        if not -1.0 <= cosine <= 1.0:
            abort(f"cosine outside [-1, 1]: {cosine}")
        expected_angle = math.degrees(math.acos(cosine))
        observed_angle = row["preupdate_angle_deg_to_liked"]
        if abs(expected_angle - observed_angle) > 1e-6:
            abort(
                "angle column is not arccos(cosine): "
                f"expected {expected_angle}, observed {observed_angle}"
            )

    rows.sort(key=lambda row: (int(row["trial"]), int(row["step"])))
    return rows


def mean_by_step(rows: list[dict], key: str) -> list[float]:
    means = []
    for step in range(1, N_STEPS + 1):
        vals = [row[key] for row in rows if int(row["step"]) == step]
        means.append(float(np.mean(vals)))
    return means


def rolling_mean(values: list[float], window: int) -> list[float | None]:
    rolled: list[float | None] = []
    for idx in range(len(values)):
        if idx + 1 < window:
            rolled.append(None)
        else:
            rolled.append(float(np.mean(values[idx + 1 - window : idx + 1])))
    return rolled


def summarize(rows: list[dict]) -> tuple[list[dict], dict]:
    summaries = []
    quartile = max(1, N_STEPS // 4)
    for trial in range(1, N_TRIALS + 1):
        trial_rows = [row for row in rows if int(row["trial"]) == trial]
        theta = [row["preupdate_angle_deg_to_liked"] for row in trial_rows]
        cosine = [row["preupdate_max_cosine_to_liked"] for row in trial_rows]
        first_theta = float(np.mean(theta[:quartile]))
        last_theta = float(np.mean(theta[-quartile:]))
        first_cosine = float(np.mean(cosine[:quartile]))
        last_cosine = float(np.mean(cosine[-quartile:]))
        summaries.append(
            {
                "trial": trial,
                "first_quartile_theta_mean": first_theta,
                "last_quartile_theta_mean": last_theta,
                "angle_ratio_last_over_first": last_theta / first_theta,
                "cosine_gain_last_over_first": last_cosine - first_cosine,
            }
        )

    aggregate = {
        "mean_first_quartile_theta": float(
            np.mean([s["first_quartile_theta_mean"] for s in summaries])
        ),
        "mean_last_quartile_theta": float(
            np.mean([s["last_quartile_theta_mean"] for s in summaries])
        ),
        "mean_angle_ratio": float(
            np.mean([s["angle_ratio_last_over_first"] for s in summaries])
        ),
        "mean_cosine_gain": float(
            np.mean([s["cosine_gain_last_over_first"] for s in summaries])
        ),
        "final_mean_relevance_at_20": mean_by_step(rows, "feed_mean_relevance_at_20")[-1],
        "final_strict_precision_at_20": mean_by_step(rows, "feed_strict_precision_at_20")[-1],
        "final_soft_precision_at_20": mean_by_step(rows, "feed_soft_precision_at_20")[-1],
    }
    return summaries, aggregate


def write_summary(summaries: list[dict], aggregate: dict) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_csv = OUT_DIR / f"{OUTPUT_PREFIX}_summary.csv"
    aggregate_csv = OUT_DIR / f"{OUTPUT_PREFIX}_aggregate.csv"

    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    with aggregate_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(aggregate))
        writer.writeheader()
        writer.writerow(aggregate)

    return summary_csv, aggregate_csv


def build_interactive_chart(rows: list[dict], summaries: list[dict], aggregate: dict) -> Path:
    steps = list(range(1, N_STEPS + 1))
    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=(
            "Pre-update theta_t to liked paper",
            "Pre-update max cosine s_t",
            "Feed mean similarity",
            "Feed mean similarity to target",
            "Feed Mean Relevance@20",
            "Strict/Soft Precision@20",
            "Raw theta_t events + rolling mean, w=10",
            "Last/first quartile angle ratio",
        ),
        vertical_spacing=0.09,
        horizontal_spacing=0.10,
    )

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
    for trial in range(1, N_TRIALS + 1):
        trial_rows = [row for row in rows if int(row["trial"]) == trial]
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=[row["preupdate_angle_deg_to_liked"] for row in trial_rows],
                mode="lines+markers",
                name=f"trial {trial}",
                line={"color": colors[trial - 1], "width": 1},
                marker={"size": 4},
                opacity=0.35,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    mean_style = {"color": "#111111", "width": 4}
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean_by_step(rows, "preupdate_angle_deg_to_liked"),
            mode="lines+markers",
            name="mean theta",
            line=mean_style,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean_by_step(rows, "preupdate_max_cosine_to_liked"),
            mode="lines+markers",
            name="mean s_t",
            line={"color": "#2B6CB0", "width": 4},
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean_by_step(rows, "feed_mean_similarity"),
            mode="lines+markers",
            name="feed mean similarity",
            line={"color": "#2F855A", "width": 4},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean_by_step(rows, "feed_mean_similarity_to_target"),
            mode="lines+markers",
            name="feed similarity to target",
            line={"color": "#805AD5", "width": 4},
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean_by_step(rows, "feed_mean_relevance_at_20"),
            mode="lines+markers",
            name="mean relevance@20",
            line={"color": "#38A169", "width": 4},
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean_by_step(rows, "feed_strict_precision_at_20"),
            mode="lines+markers",
            name="strict P@20",
            line={"color": "#E45756", "width": 3},
        ),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean_by_step(rows, "feed_soft_precision_at_20"),
            mode="lines+markers",
            name="soft P@20",
            line={"color": "#4C78A8", "width": 3},
        ),
        row=3,
        col=2,
    )

    flat_theta = [row["preupdate_angle_deg_to_liked"] for row in rows]
    event_x = list(range(1, len(flat_theta) + 1))
    fig.add_trace(
        go.Scatter(
            x=event_x,
            y=flat_theta,
            mode="markers",
            name="raw theta events",
            marker={"size": 5, "color": "#4C78A8", "opacity": 0.5},
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=event_x,
            y=rolling_mean(flat_theta, ROLLING_WINDOW),
            mode="lines",
            name="rolling mean theta",
            line={"color": "#111111", "width": 4},
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[f"T{s['trial']}" for s in summaries],
            y=[s["angle_ratio_last_over_first"] for s in summaries],
            name="last/first theta ratio",
            marker_color="#4C78A8",
        ),
        row=4,
        col=2,
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="#888888", row=4, col=2)
    fig.add_hline(
        y=aggregate["mean_angle_ratio"],
        line_dash="dot",
        line_color="#111111",
        annotation_text=f"mean={aggregate['mean_angle_ratio']:.3f}",
        row=4,
        col=2,
    )

    fig.update_yaxes(title_text="degrees; lower is better", row=1, col=1)
    fig.update_yaxes(title_text="cosine; higher is better", row=1, col=2)
    fig.update_yaxes(title_text="similarity", row=2, col=1)
    fig.update_yaxes(title_text="target similarity", row=2, col=2)
    fig.update_yaxes(title_text="0-1", row=3, col=1)
    fig.update_yaxes(title_text="0-1", row=3, col=2)
    fig.update_yaxes(title_text="degrees", row=4, col=1)
    fig.update_yaxes(title_text="ratio; lower is better", row=4, col=2)
    for row in range(1, 4):
        for col in range(1, 3):
            fig.update_xaxes(title_text="interaction step", row=row, col=col)
    fig.update_xaxes(title_text="positive feedback event", row=4, col=1)
    fig.update_xaxes(title_text="trial", row=4, col=2)

    fig.update_layout(
        title=CHART_TITLE,
        width=1500,
        height=1500,
        template="plotly_white",
        legend={"orientation": "h", "y": -0.05},
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = OUT_DIR / f"{OUTPUT_PREFIX}_trajectory.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    return html_path


def build_static_png(rows: list[dict], summaries: list[dict], aggregate: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 1800, 1350
    margin = 80
    gap_x = 80
    gap_y = 95
    panel_w = (width - 2 * margin - gap_x) // 2
    panel_h = (height - 2 * margin - 2 * gap_y) // 3
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    def bounds(row: int, col: int) -> tuple[int, int, int, int]:
        x0 = margin + col * (panel_w + gap_x)
        y0 = margin + row * (panel_h + gap_y)
        return x0, y0, x0 + panel_w, y0 + panel_h

    def panel(
        row: int,
        col: int,
        title: str,
        series: list[tuple[str, list[float], str]],
        y_min: float | None = None,
        y_max: float | None = None,
    ) -> None:
        x0, y0, x1, y1 = bounds(row, col)
        px0, py0, px1, py1 = x0 + 65, y0 + 34, x1 - 20, y1 - 42
        draw.text((x0, y0), title, fill="#111111", font=font)
        values = [value for _, vals, _ in series for value in vals]
        lo = min(values) if y_min is None else y_min
        hi = max(values) if y_max is None else y_max
        if abs(hi - lo) < 1e-9:
            hi = lo + 1.0
        draw.rectangle((px0, py0, px1, py1), outline="#D0D0D0")
        for idx in range(5):
            y = py1 - idx * (py1 - py0) / 4
            value = lo + idx * (hi - lo) / 4
            draw.line((px0, y, px1, y), fill="#EEEEEE")
            draw.text((x0, y - 6), f"{value:.3f}", fill="#555555", font=font)
        for step in (1, 4, 8, 12):
            x = px0 + (step - 1) * (px1 - px0) / (N_STEPS - 1)
            draw.text((x - 8, py1 + 10), str(step), fill="#555555", font=font)
        for s_idx, (name, vals, color) in enumerate(series):
            points = []
            for i, val in enumerate(vals):
                x = px0 + i * (px1 - px0) / (len(vals) - 1)
                y = py1 - (val - lo) * (py1 - py0) / (hi - lo)
                points.append((x, y))
            draw.line(points, fill=color, width=4)
            for x, y in points:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
            draw.text((px0 + 170 * s_idx, y1 - 24), name, fill=color, font=font)

    panel(
        0,
        0,
        "Pre-update theta_t to liked paper (deg, lower is better)",
        [("mean theta", mean_by_step(rows, "preupdate_angle_deg_to_liked"), "#111111")],
    )
    panel(
        0,
        1,
        "Pre-update nearest-centroid cosine s_t",
        [("mean s_t", mean_by_step(rows, "preupdate_max_cosine_to_liked"), "#2B6CB0")],
    )
    panel(
        1,
        0,
        "Feed similarity metrics",
        [
            ("feed sim", mean_by_step(rows, "feed_mean_similarity"), "#2F855A"),
            ("target sim", mean_by_step(rows, "feed_mean_similarity_to_target"), "#805AD5"),
        ],
    )
    panel(
        1,
        1,
        "Feed relevance / precision metrics",
        [
            ("rel@20", mean_by_step(rows, "feed_mean_relevance_at_20"), "#38A169"),
            ("soft P@20", mean_by_step(rows, "feed_soft_precision_at_20"), "#4C78A8"),
        ],
        y_min=0,
        y_max=1.0,
    )
    panel(
        2,
        0,
        "Picked-paper target similarity",
        [
            (
                "picked target sim",
                mean_by_step(rows, "picked_paper_target_similarity"),
                "#E45756",
            )
        ],
    )

    x0, y0, x1, y1 = bounds(2, 1)
    draw.text((x0, y0), "Angle ratio by trial: last quartile / first quartile", fill="#111111", font=font)
    px0, py0, px1, py1 = x0 + 65, y0 + 34, x1 - 20, y1 - 42
    draw.rectangle((px0, py0, px1, py1), outline="#D0D0D0")
    ratios = [summary["angle_ratio_last_over_first"] for summary in summaries]
    ratio_pad = 0.02
    lo = max(0.0, min(ratios) - ratio_pad)
    hi = max(1.01, max(ratios) + ratio_pad)
    for idx in range(4):
        y = py1 - idx * (py1 - py0) / 3
        value = lo + idx * (hi - lo) / 3
        draw.line((px0, y, px1, y), fill="#EEEEEE")
        draw.text((x0, y - 6), f"{value:.3f}", fill="#555555", font=font)
    for idx, ratio in enumerate(ratios):
        bar_w = (px1 - px0) / (len(ratios) * 1.7)
        x = px0 + idx * (px1 - px0) / len(ratios) + bar_w * 0.35
        top = py1 - (ratio - lo) * (py1 - py0) / (hi - lo)
        baseline = py1 - (1.0 - lo) * (py1 - py0) / (hi - lo)
        y_a, y_b = sorted((top, baseline))
        draw.rectangle((x, y_a, x + bar_w, y_b), fill="#4C78A8")
        draw.text((x - 1, py1 + 10), f"T{idx + 1}", fill="#555555", font=font)
    mean_y = py1 - (aggregate["mean_angle_ratio"] - lo) * (py1 - py0) / (hi - lo)
    draw.line((px0, mean_y, px1, mean_y), fill="#111111", width=3)
    draw.text((px0 + 6, mean_y + 5), f"mean={aggregate['mean_angle_ratio']:.3f}", fill="#111111", font=font)

    draw.text(
        (margin, 24),
        STATIC_TITLE,
        fill="#111111",
        font=font,
    )
    png_path = OUT_DIR / f"{OUTPUT_PREFIX}_trajectory.png"
    image.save(png_path)
    return png_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rows = load_and_validate(INPUT_CSV)
    summaries, aggregate = summarize(rows)
    summary_csv, aggregate_csv = write_summary(summaries, aggregate)
    html_path = build_interactive_chart(rows, summaries, aggregate)
    png_path = build_static_png(rows, summaries, aggregate)

    print("Uploaded Angle-to-Liked evaluation")
    print(f"input_csv: {INPUT_CSV}")
    print(f"rows: {len(rows)}")
    print(f"trials: {N_TRIALS}")
    print(f"steps_per_trial: {N_STEPS}")
    for key, value in aggregate.items():
        print(f"{key}: {value:.6f}")
    print(f"summary_csv: {summary_csv}")
    print(f"aggregate_csv: {aggregate_csv}")
    print(f"chart_html: {html_path}")
    print(f"chart_png: {png_path}")


if __name__ == "__main__":
    main()
