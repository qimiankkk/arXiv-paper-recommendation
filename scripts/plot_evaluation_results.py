"""Generate presentation-ready evaluation figures for arXiv recommendations.

Required query-only inputs:
    --query-aggregate: aggregate metrics JSON with ``overall_average`` and
        optional per-evaluation-case entries.

Optional query inputs:
    --query-eval: detailed per-paper evaluation JSON. Used to compute query
        scores when needed by downstream experiments.
    --query-scores: score-by-rank JSON used for the relevance-by-rank curve.
    --query-baseline-comparison: baseline-vs-profile metrics JSON.

Optional daily-feed inputs:
    --daily-variant-metrics: daily-feed variant metrics JSON.
    --diversity-sweep: diversity-index sweep metrics JSON.

Example:
    python scripts/plot_evaluation_results.py \\
      --query-aggregate evaluation/data/query_aggregate_metrics_combined_ev1_ev120.json \\
      --out-dir reports/eval_figures
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

K = 20
FIG_DPI = 300
QUERY_METRICS = (
    "strict_precision@20",
    "soft_precision@20",
    "mean_relevance@20",
    "ndcg@20",
)
DAILY_FEED_VARIANT_ORDER = (
    "single_vector_average_baseline",
    "multi_centroid_no_diversity",
    "full_daily_feed",
)
DAILY_FEED_VARIANT_LABELS = {
    "single_vector_average_baseline": "Single-vector baseline",
    "multi_centroid_no_diversity": "Multi-centroid",
    "full_daily_feed": "Full Daily Feed",
}
DAILY_FEED_RELEVANCE_METRICS = QUERY_METRICS
DAILY_FEED_DIVERSITY_METRICS = (
    "thread_coverage@20",
    "redundancy@20",
)
DAILY_FEED_METRIC_LABELS = {
    "strict_precision@20": "Strict Precision@20",
    "soft_precision@20": "Soft Precision@20",
    "mean_relevance@20": "Mean Relevance@20",
    "ndcg@20": "NDCG@20",
    "thread_coverage@20": "Thread Coverage@20",
    "redundancy@20": "Redundancy@20",
}
DAILY_FEED_COLORS = {
    "strict_precision@20": "#93c5fd",
    "soft_precision@20": "#86efac",
    "mean_relevance@20": "#fdba74",
    "ndcg@20": "#c4b5fd",
    "thread_coverage@20": "#93c5fd",
    "redundancy@20": "#fdba74",
    "single_vector_average_baseline": "#64748b",
    "multi_centroid_no_diversity": "#2563eb",
    "full_daily_feed": "#0f766e",
}
HEADLINE_COLUMNS = {
    "strict_precision@20": "Strict P@20",
    "soft_precision@20": "Soft P@20",
    "mean_relevance@20": "Mean Rel@20",
    "ndcg@20": "NDCG@20",
    "thread_coverage@20": "Coverage@20",
    "coverage@20": "Coverage@20",
    "redundancy@20": "Redundancy@20",
}
HEADLINE_TABLE_SPECS = (
    ("Strict P@20", ("strict_precision@20",)),
    ("Soft P@20", ("soft_precision@20",)),
    ("Mean Rel@20", ("mean_relevance@20",)),
    ("NDCG@20", ("ndcg@20",)),
    ("Coverage@20", ("thread_coverage@20", "coverage@20")),
    ("Redundancy@20", ("redundancy@20",)),
)


def _require_matplotlib():
    try:
        if "MPLCONFIGDIR" not in os.environ:
            cache_dir = Path(tempfile.gettempdir()) / "arxiv_eval_matplotlib"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(cache_dir)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to generate evaluation figures. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return plt


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required to write evaluation tables. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return pd


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "numpy is required to summarize evaluation metrics. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return np


def _style_axes(ax) -> None:
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(colors="#374151")


def _display_name(name: str) -> str:
    return str(name).replace("_", " ").replace("-", " ").title()


def _sort_eval_ids(eval_ids: list[str]) -> list[str]:
    def key(value: str) -> tuple[int, int | str]:
        text = str(value)
        if text.startswith("ev") and text[2:].isdigit():
            return (0, int(text[2:]))
        return (1, text)

    return sorted(eval_ids, key=key)


def _round_or_blank(value: Any) -> float | str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return round(number, 3)


def _standard_error(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    np = _require_numpy()
    return float(np.std(np.asarray(values, dtype=float), ddof=1) / math.sqrt(len(values)))


def _save_figure_png_pdf(fig, out_dir: Path, stem: str) -> list[Path]:
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return [png_path, pdf_path]


def load_json(path: str | Path) -> Any:
    """Load strict JSON, raising a path-aware error for malformed files."""
    json_path = Path(path)
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {json_path}: {exc.msg} at line {exc.lineno}, "
            f"column {exc.colno}. Fix JSON escaping before plotting. "
            'A common issue is an unescaped title such as: '
            'What Do Machine Learning Researchers Mean by "Reproducible"?'
        ) from exc


def compute_query_scores_from_eval_json(path: str | Path) -> dict[str, list[float]]:
    """Compute weighted query relevance scores grouped by eval set.

    Input rows are assumed to be in ranked order within each eval set.
    """
    payload = load_json(path)
    rows = payload.get("evaluations") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain an 'evaluations' list.")

    scores_by_eval_set: dict[str, list[float]] = defaultdict(list)
    for i, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Evaluation row {i} in {path} is not an object.")
        eval_set = row.get("eval_set")
        if not eval_set:
            raise ValueError(f"Evaluation row {i} in {path} is missing eval_set.")
        try:
            score = (
                0.60 * float(row["query_relevance"])
                + 0.30 * float(row["user_interest_relevance"])
                + 0.10 * float(row["usefulness"])
            )
        except KeyError as exc:
            raise ValueError(f"Evaluation row {i} in {path} is missing {exc.args[0]!r}.") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Evaluation row {i} in {path} has a non-numeric score field.") from exc
        scores_by_eval_set[str(eval_set)].append(round(float(score), 10))

    return dict(scores_by_eval_set)


def _ndcg(scores: list[float]) -> float:
    if not scores:
        return 0.0
    dcg = sum(score / math.log2(rank + 1) for rank, score in enumerate(scores, start=1))
    ideal = sorted(scores, reverse=True)
    idcg = sum(score / math.log2(rank + 1) for rank, score in enumerate(ideal, start=1))
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def _metrics_for_scores(scores: list[float]) -> dict[str, float]:
    top_scores = [float(score) for score in scores[:K]]
    if not top_scores:
        return {metric: 0.0 for metric in QUERY_METRICS}
    n = len(top_scores)
    return {
        "strict_precision@20": sum(score >= 1.5 for score in top_scores) / n,
        "soft_precision@20": sum(score >= 1.0 for score in top_scores) / n,
        "mean_relevance@20": (sum(top_scores) / n) / 2.0,
        "ndcg@20": _ndcg(top_scores),
    }


def compute_metrics(scores_by_eval_set: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Compute per-case and overall query metrics from weighted scores."""
    metrics: dict[str, dict[str, float]] = {}
    for eval_set in _sort_eval_ids(list(scores_by_eval_set)):
        metrics[eval_set] = _metrics_for_scores(scores_by_eval_set[eval_set])

    if metrics:
        metrics["overall_average"] = {
            metric: sum(case[metric] for case in metrics.values()) / len(metrics)
            for metric in QUERY_METRICS
        }
    else:
        metrics["overall_average"] = {metric: 0.0 for metric in QUERY_METRICS}
    return metrics


def _headline_rows(
    query_aggregate: dict[str, Any] | None = None,
    daily_variant_metrics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if query_aggregate:
        overall = query_aggregate.get("overall_average")
        if not isinstance(overall, dict):
            raise ValueError("Query aggregate metrics must include an overall_average object.")
        row = {"Algorithm": "Query Search"}
        row.update(overall)
        rows.append(row)

    if daily_variant_metrics:
        if "full_daily_feed" in daily_variant_metrics:
            daily_name = "full_daily_feed"
        else:
            daily_name = next(iter(daily_variant_metrics))
        daily = daily_variant_metrics[daily_name]
        if not isinstance(daily, dict):
            raise ValueError("Daily variant metric entries must be objects.")
        row = {"Algorithm": "Daily Feed"}
        row.update(daily)
        rows.append(row)
    return rows


def plot_headline_table(rows: list[dict[str, Any]], out_dir: str | Path) -> Any:
    """Write headline metrics as CSV and PNG table."""
    if not rows:
        raise ValueError("No headline rows are available to plot.")

    pd = _require_pandas()
    plt = _require_matplotlib()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    table_rows: list[dict[str, Any]] = []
    for row in rows:
        table_row = {"Algorithm": row["Algorithm"]}
        for label, metric_keys in HEADLINE_TABLE_SPECS:
            value = None
            for metric in metric_keys:
                if metric in row:
                    value = row[metric]
                    break
            table_row[label] = _round_or_blank(value)
        table_rows.append(table_row)

    df = pd.DataFrame(table_rows)
    csv_path = out_path / "headline_metrics_table.csv"
    png_path = out_path / "headline_metrics_table.png"
    df.to_csv(csv_path, index=False)

    fig_width = max(8.0, 1.35 * len(df.columns))
    fig_height = max(1.8, 0.6 * (len(df) + 1))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=df.astype(str).values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (row_i, _col_i), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if row_i == 0:
            cell.set_facecolor("#f3f4f6")
            cell.set_text_props(weight="bold", color="#111827")
        else:
            cell.set_facecolor("#ffffff")
    fig.tight_layout()
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return df


def plot_query_case_metrics(query_aggregate: dict[str, Any], out_dir: str | Path) -> Path | None:
    """Plot query metrics for each evaluation case."""
    cases = {
        key: value
        for key, value in query_aggregate.items()
        if key != "overall_average" and isinstance(value, dict)
    }
    if not cases:
        print("Skipping query_search_case_metrics.png: no per-case query metrics found.")
        return None

    plt = _require_matplotlib()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    case_ids = _sort_eval_ids(list(cases))
    x = list(range(len(case_ids)))

    fig, axes = plt.subplots(
        len(QUERY_METRICS),
        1,
        figsize=(max(10.0, len(case_ids) * 0.11), 8.5),
        sharex=True,
    )
    for ax, metric in zip(axes, QUERY_METRICS):
        values = [float(cases[case_id].get(metric, 0.0)) for case_id in case_ids]
        ax.plot(x, values, marker="o", markersize=2.5, linewidth=1.2, color="#2563eb")
        ax.set_ylabel(HEADLINE_COLUMNS[metric])
        ax.set_ylim(0, max(1.05, max(values) * 1.1 if values else 1.0))
        _style_axes(ax)

    tick_step = max(1, math.ceil(len(case_ids) / 24))
    shown_ticks = x[::tick_step]
    axes[-1].set_xticks(shown_ticks)
    axes[-1].set_xticklabels([case_ids[i] for i in shown_ticks], rotation=45, ha="right")
    axes[-1].set_xlabel("Evaluation Case")
    fig.suptitle("Query Search Metrics by Evaluation Case", y=0.995, fontsize=13, weight="bold")
    fig.tight_layout()
    output = out_path / "query_search_case_metrics.png"
    fig.savefig(output, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return output


def _rank_number(rank_key: str) -> int | None:
    text = str(rank_key)
    if text.startswith("p") and text[1:].isdigit():
        return int(text[1:])
    if text.isdigit():
        return int(text)
    return None


def plot_relevance_by_rank(query_scores: dict[str, Any], out_dir: str | Path) -> Any:
    """Write average relevance-by-rank CSV and PNG from score-by-rank JSON."""
    rank_values: dict[int, list[float]] = {rank: [] for rank in range(1, K + 1)}
    for eval_set, scores in query_scores.items():
        if not isinstance(scores, dict):
            raise ValueError(f"Scores for {eval_set} must be an object keyed by rank.")
        for key, value in scores.items():
            rank = _rank_number(key)
            if rank is None or rank < 1 or rank > K:
                continue
            try:
                rank_values[rank].append(float(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Score {key} for {eval_set} is not numeric.") from exc

    rows = []
    for rank in range(1, K + 1):
        values = rank_values[rank]
        avg = sum(values) / len(values) if values else None
        rows.append({"rank": rank, "avg_score": _round_or_blank(avg)})

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pd = _require_pandas()
    df = pd.DataFrame(rows)
    df.to_csv(out_path / "query_relevance_by_rank.csv", index=False)

    plt = _require_matplotlib()
    plot_df = df[df["avg_score"] != ""].copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(plot_df["rank"], plot_df["avg_score"], marker="o", linewidth=1.8, color="#2563eb")
    ax.set_title("Query Search Relevance by Rank", weight="bold")
    ax.set_xlabel("Rank Position")
    ax.set_ylabel("Average Paper-Level Score")
    ax.set_xlim(1, K)
    ax.set_ylim(0, 2)
    ax.set_xticks(range(1, K + 1))
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path / "query_relevance_by_rank.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return df


def plot_baseline_comparison(baseline_metrics: dict[str, Any], out_dir: str | Path) -> list[Path]:
    """Plot query baseline comparisons for NDCG and mean relevance."""
    plt = _require_matplotlib()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    specs = (
        ("ndcg@20", "NDCG@20", "query_baseline_comparison_ndcg.png"),
        ("mean_relevance@20", "Mean Rel@20", "query_baseline_comparison_mean_relevance.png"),
    )
    labels = [_display_name(name) for name in baseline_metrics]

    for metric, ylabel, filename in specs:
        values = []
        for name, metrics in baseline_metrics.items():
            if not isinstance(metrics, dict):
                raise ValueError(f"Baseline entry {name} must be an object.")
            values.append(float(metrics.get(metric, 0.0)))
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.bar(labels, values, color=["#64748b", "#2563eb", "#0f766e", "#9333ea"][: len(labels)])
        ax.set_title(f"Query Baseline Comparison: {ylabel}", weight="bold")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(1.0, max(values) * 1.15 if values else 1.0))
        ax.tick_params(axis="x", rotation=20)
        _style_axes(ax)
        fig.tight_layout()
        output = out_path / filename
        fig.savefig(output, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)
    return outputs


def plot_daily_variant_comparison(daily_metrics: dict[str, Any], out_dir: str | Path) -> Path:
    """Plot grouped bars for daily-feed variant metrics."""
    plt = _require_matplotlib()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metrics = ("thread_coverage@20", "mean_relevance@20", "redundancy@20")
    variants = list(daily_metrics)
    x = list(range(len(variants)))
    width = 0.24

    fig, ax = plt.subplots(figsize=(max(8.5, len(variants) * 1.4), 5.2))
    colors = ("#617bb4", "#709E9A", "#B77F57")
    for offset_i, metric in enumerate(metrics):
        values = [float(daily_metrics[name].get(metric, 0.0)) for name in variants]
        positions = [i + (offset_i - 1) * width for i in x]
        ax.bar(positions, values, width=width, label=HEADLINE_COLUMNS[metric], color=colors[offset_i])
    ax.set_title("Daily Feed Variant Comparison", weight="bold")
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([_display_name(name) for name in variants], rotation=25, ha="right")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    _style_axes(ax)
    fig.tight_layout()
    output = out_path / "daily_feed_variant_comparison.png"
    fig.savefig(output, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_diversity_sweep(sweep_metrics: dict[str, Any], out_dir: str | Path) -> Path:
    """Plot diversity-index sweep metrics with numeric delta keys."""
    plt = _require_matplotlib()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    delta_items = sorted((float(key), value) for key, value in sweep_metrics.items())
    deltas = [delta for delta, _value in delta_items]
    metrics = ("thread_coverage@20", "mean_relevance@20", "redundancy@20")
    colors = ("#617bb4", "#709E9A", "#B77F57")

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for metric, color in zip(metrics, colors):
        values = [float(metric_values.get(metric, 0.0)) for _delta, metric_values in delta_items]
        ax.plot(deltas, values, marker="o", linewidth=1.8, label=HEADLINE_COLUMNS[metric], color=color)
    ax.set_title("Diversity Index Sweep", weight="bold")
    ax.set_xlabel("Diversity Index (delta)")
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    output = out_path / "diversity_index_sweep.png"
    fig.savefig(output, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return output


def _daily_feed_cases_from_payload(payload: dict[str, Any], path: str | Path) -> list[dict[str, Any]]:
    evaluations = payload.get("evaluations") if isinstance(payload, dict) else None
    if isinstance(evaluations, dict):
        cases = []
        for eval_id, case in evaluations.items():
            if not isinstance(case, dict):
                raise ValueError(f"Daily Feed evaluation {eval_id} in {path} is not an object.")
            normalized = dict(case)
            normalized.setdefault("eval_id", str(eval_id))
            cases.append(normalized)
        return cases
    if isinstance(evaluations, list):
        cases = []
        for i, case in enumerate(evaluations, start=1):
            if not isinstance(case, dict):
                raise ValueError(f"Daily Feed evaluation row {i} in {path} is not an object.")
            normalized = dict(case)
            normalized.setdefault("eval_id", f"ev{i}")
            cases.append(normalized)
        return cases
    raise ValueError(f"{path} must contain a top-level 'evaluations' object or list.")


def load_daily_feed_evaluation_cases(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate raw Daily Feed evaluation cases."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Daily Feed evaluation input does not exist: {input_path}")
    cases = _daily_feed_cases_from_payload(load_json(input_path), input_path)
    if not cases:
        raise ValueError(f"{input_path} does not contain any Daily Feed evaluation cases.")
    return cases


def filter_daily_feed_cases(
    cases: list[dict[str, Any]],
    *,
    delta: float = 0.5,
    pool_deltas: bool = False,
) -> list[dict[str, Any]]:
    """Filter Daily Feed cases by delta, validating required case fields."""
    filtered: list[dict[str, Any]] = []
    for i, case in enumerate(cases, start=1):
        eval_id = case.get("eval_id", f"case {i}")
        if "algorithm_variant" not in case:
            raise ValueError(f"Daily Feed {eval_id} is missing algorithm_variant.")
        if "diversity_index_delta" not in case:
            raise ValueError(f"Daily Feed {eval_id} is missing diversity_index_delta.")
        try:
            case_delta = float(case["diversity_index_delta"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Daily Feed {eval_id} has non-numeric diversity_index_delta.") from exc
        if pool_deltas or math.isclose(case_delta, float(delta), abs_tol=1e-9):
            filtered.append(case)
    if not filtered:
        selector = "all deltas" if pool_deltas else f"delta={delta}"
        raise ValueError(f"No Daily Feed evaluation cases matched {selector}.")
    _validate_daily_feed_cases_for_plots(filtered)
    return filtered


def _validate_daily_feed_cases_for_plots(cases: list[dict[str, Any]]) -> None:
    required_metrics = set(DAILY_FEED_RELEVANCE_METRICS + DAILY_FEED_DIVERSITY_METRICS)
    variants = {str(case["algorithm_variant"]) for case in cases}
    missing_variants = [variant for variant in DAILY_FEED_VARIANT_ORDER if variant not in variants]
    if missing_variants:
        raise ValueError(
            "Filtered Daily Feed cases are missing required variants: "
            + ", ".join(missing_variants)
        )

    for case in cases:
        eval_id = case.get("eval_id", "<unknown>")
        variant = case.get("algorithm_variant")
        if variant not in DAILY_FEED_VARIANT_ORDER:
            raise ValueError(f"Daily Feed {eval_id} has unsupported variant {variant!r}.")
        metrics = case.get("computed_metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"Daily Feed {eval_id} is missing computed_metrics.")
        for metric in required_metrics:
            if metric not in metrics:
                raise ValueError(f"Daily Feed {eval_id} computed_metrics is missing {metric}.")
            try:
                float(metrics[metric])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Daily Feed {eval_id} metric {metric} is not numeric.") from exc

        papers = case.get("papers")
        if not isinstance(papers, list):
            raise ValueError(f"Daily Feed {eval_id} is missing a papers list.")
        if len(papers) != K:
            print(f"Warning: Daily Feed {eval_id} has {len(papers)} papers, expected {K}.")
        for paper_i, paper in enumerate(papers, start=1):
            if not isinstance(paper, dict):
                raise ValueError(f"Daily Feed {eval_id} paper {paper_i} is not an object.")
            if "rank" not in paper or "r_feed" not in paper:
                raise ValueError(f"Daily Feed {eval_id} paper {paper_i} is missing rank or r_feed.")
            try:
                int(paper["rank"])
                float(paper["r_feed"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Daily Feed {eval_id} paper {paper_i} has non-numeric rank or r_feed.") from exc


def summarize_daily_feed_metric_cases(
    cases: list[dict[str, Any]],
    metrics: tuple[str, ...],
    *,
    delta_label: str,
) -> Any:
    """Summarize per-case Daily Feed metrics by variant with standard errors."""
    pd = _require_pandas()
    rows: list[dict[str, Any]] = []
    for variant in DAILY_FEED_VARIANT_ORDER:
        variant_cases = [case for case in cases if case["algorithm_variant"] == variant]
        for metric in metrics:
            values = [float(case["computed_metrics"][metric]) for case in variant_cases]
            rows.append(
                {
                    "algorithm_variant": variant,
                    "algorithm_variant_label": DAILY_FEED_VARIANT_LABELS[variant],
                    "metric": metric,
                    "metric_label": DAILY_FEED_METRIC_LABELS[metric],
                    "mean": round(sum(values) / len(values), 6),
                    "standard_error": round(_standard_error(values), 6),
                    "n": len(values),
                    "delta_filter": delta_label,
                }
            )
    return pd.DataFrame(rows)


def summarize_daily_feed_relevance_by_rank(cases: list[dict[str, Any]], *, delta_label: str) -> Any:
    """Summarize normalized r_feed by rank and variant."""
    pd = _require_pandas()
    rows: list[dict[str, Any]] = []
    for variant in DAILY_FEED_VARIANT_ORDER:
        variant_cases = [case for case in cases if case["algorithm_variant"] == variant]
        for rank in range(1, K + 1):
            values: list[float] = []
            for case in variant_cases:
                for paper in case["papers"]:
                    if int(paper["rank"]) == rank:
                        values.append(float(paper["r_feed"]) / 2.0)
                        break
            rows.append(
                {
                    "algorithm_variant": variant,
                    "algorithm_variant_label": DAILY_FEED_VARIANT_LABELS[variant],
                    "rank": rank,
                    "mean_normalized_r_feed": round(sum(values) / len(values), 6) if values else "",
                    "standard_error": round(_standard_error(values), 6) if values else "",
                    "n": len(values),
                    "delta_filter": delta_label,
                }
            )
    return pd.DataFrame(rows)


def _plot_daily_feed_grouped_metric_bars(
    summary_df: Any,
    metrics: tuple[str, ...],
    *,
    title: str,
    caption: str,
    output_stem: str,
    out_dir: str | Path,
) -> list[Path]:
    plt = _require_matplotlib()
    np = _require_numpy()
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(DAILY_FEED_VARIANT_ORDER), dtype=float)
    width = min(0.8 / len(metrics), 0.28)
    fig, ax = plt.subplots(figsize=(10.8, 6.4), constrained_layout=False)

    for metric_i, metric in enumerate(metrics):
        metric_df = summary_df[summary_df["metric"] == metric].set_index("algorithm_variant")
        values = [float(metric_df.loc[variant, "mean"]) for variant in DAILY_FEED_VARIANT_ORDER]
        errors = [float(metric_df.loc[variant, "standard_error"]) for variant in DAILY_FEED_VARIANT_ORDER]
        positions = x + (metric_i - (len(metrics) - 1) / 2) * width
        ax.bar(
            positions,
            values,
            width=width,
            yerr=errors,
            capsize=3,
            label=DAILY_FEED_METRIC_LABELS[metric],
            color=DAILY_FEED_COLORS[metric],
            edgecolor="#ffffff",
            linewidth=1.0,
        )

    ax.set_title(title, fontsize=15, weight="bold", pad=60)
    ax.text(0.5, 1.11, caption, transform=ax.transAxes, ha="center", va="bottom", color="#4b5563", fontsize=10.5)
    ax.set_ylabel("Metric value", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [DAILY_FEED_VARIANT_LABELS[variant] for variant in DAILY_FEED_VARIANT_ORDER],
        rotation=0,
        ha="center",
        fontsize=12.5,
    )
    ax.legend(
        frameon=False,
        ncols=min(2, len(metrics)),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        fontsize=10.5,
        columnspacing=1.4,
        handlelength=1.2,
    )
    _style_axes(ax)
    fig.subplots_adjust(top=0.72, bottom=0.14, left=0.08, right=0.98)
    outputs = _save_figure_png_pdf(fig, output_dir, output_stem)
    plt.close(fig)
    return outputs


def plot_daily_feed_relevance_metrics_by_variant(
    cases: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    delta_label: str,
) -> list[Path]:
    """Create Daily Feed relevance metric grouped bars and summary CSV."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_daily_feed_metric_cases(cases, DAILY_FEED_RELEVANCE_METRICS, delta_label=delta_label)
    csv_path = output_dir / "daily_feed_relevance_metrics_by_variant.csv"
    summary.to_csv(csv_path, index=False)
    caption = f"{delta_label}, averaged over user profiles"
    outputs = _plot_daily_feed_grouped_metric_bars(
        summary,
        DAILY_FEED_RELEVANCE_METRICS,
        title="Daily Feed Relevance Metrics by Variant",
        caption=caption,
        output_stem="daily_feed_relevance_metrics_by_variant",
        out_dir=output_dir,
    )
    return outputs + [csv_path]


def plot_daily_feed_diversity_metrics_by_variant(
    cases: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    delta_label: str,
) -> list[Path]:
    """Create Daily Feed diversity metric grouped bars and summary CSV."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_daily_feed_metric_cases(cases, DAILY_FEED_DIVERSITY_METRICS, delta_label=delta_label)
    csv_path = output_dir / "daily_feed_diversity_metrics_by_variant.csv"
    summary.to_csv(csv_path, index=False)
    caption = "Higher Thread Coverage is better; lower Redundancy is better."
    outputs = _plot_daily_feed_grouped_metric_bars(
        summary,
        DAILY_FEED_DIVERSITY_METRICS,
        title="Daily Feed Diversity Metrics by Variant",
        caption=caption,
        output_stem="daily_feed_diversity_metrics_by_variant",
        out_dir=output_dir,
    )
    return outputs + [csv_path]


def plot_daily_feed_relevance_by_rank_by_variant(
    cases: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    delta_label: str,
) -> list[Path]:
    """Create Daily Feed relevance-by-rank lines and summary CSV."""
    plt = _require_matplotlib()
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_daily_feed_relevance_by_rank(cases, delta_label=delta_label)
    csv_path = output_dir / "daily_feed_relevance_by_rank_by_variant.csv"
    summary.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
    for variant in DAILY_FEED_VARIANT_ORDER:
        variant_df = summary[summary["algorithm_variant"] == variant]
        x = [int(rank) for rank in variant_df["rank"]]
        y = [float(value) for value in variant_df["mean_normalized_r_feed"]]
        err = [float(value) for value in variant_df["standard_error"]]
        color = DAILY_FEED_COLORS[variant]
        ax.plot(
            x,
            y,
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=DAILY_FEED_VARIANT_LABELS[variant],
            color=color,
        )
        lower = [max(0.0, yi - ei) for yi, ei in zip(y, err)]
        upper = [min(1.0, yi + ei) for yi, ei in zip(y, err)]
        ax.fill_between(x, lower, upper, color=color, alpha=0.12, linewidth=0)

    ax.set_title("Daily Feed Relevance by Rank", fontsize=14, weight="bold", pad=14)
    ax.set_xlabel("Rank position")
    ax.set_ylabel("Average normalized r_feed")
    ax.set_xlim(1, K)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(1, K + 1))
    ax.legend(frameon=False)
    _style_axes(ax)
    outputs = _save_figure_png_pdf(fig, output_dir, "daily_feed_relevance_by_rank_by_variant")
    plt.close(fig)
    return outputs + [csv_path]


def run_daily_feed_evaluation_plots(
    input_path: str | Path,
    out_dir: str | Path,
    *,
    delta: float = 0.5,
    pool_deltas: bool = False,
) -> dict[str, Any]:
    """Generate the three presentation Daily Feed evaluation plots."""
    cases = load_daily_feed_evaluation_cases(input_path)
    used_cases = filter_daily_feed_cases(cases, delta=delta, pool_deltas=pool_deltas)
    delta_label = "pooled deltas" if pool_deltas else f"delta = {delta:g}"
    variants_found = [variant for variant in DAILY_FEED_VARIANT_ORDER if any(c["algorithm_variant"] == variant for c in used_cases)]

    output_paths: list[Path] = []
    output_paths.extend(plot_daily_feed_relevance_metrics_by_variant(used_cases, out_dir, delta_label=delta_label))
    output_paths.extend(plot_daily_feed_diversity_metrics_by_variant(used_cases, out_dir, delta_label=delta_label))
    output_paths.extend(plot_daily_feed_relevance_by_rank_by_variant(used_cases, out_dir, delta_label=delta_label))

    print(f"Loaded Daily Feed evaluation cases: {len(cases)}")
    print(f"Used cases after filtering: {len(used_cases)}")
    print("Variants found: " + ", ".join(variants_found))
    print("Wrote Daily Feed evaluation outputs:")
    for path in output_paths:
        print(f"  {path}")

    return {
        "loaded_cases": len(cases),
        "used_cases": len(used_cases),
        "variants_found": variants_found,
        "output_paths": output_paths,
    }


def parse_daily_feed_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Daily Feed evaluation plots. Variant comparison defaults to "
            "delta=0.5; use --pool-deltas only for exploratory plots."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw Daily Feed evaluation JSON.")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory for figures and summary CSVs.")
    parser.add_argument("--delta", type=float, default=0.5, help="Diversity index delta to compare; default 0.5.")
    parser.add_argument("--pool-deltas", action="store_true", help="Pool all delta values for exploratory plots.")
    return parser.parse_args(argv)


def main_daily_feed(argv: list[str] | None = None) -> int:
    args = parse_daily_feed_args(argv)
    run_daily_feed_evaluation_plots(
        input_path=args.input,
        out_dir=args.outdir,
        delta=args.delta,
        pool_deltas=args.pool_deltas,
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-eval", type=Path, help="Detailed query evaluation JSON.")
    parser.add_argument("--query-aggregate", type=Path, help="Aggregate query metrics JSON.")
    parser.add_argument("--query-scores", type=Path, help="Score-by-rank query JSON.")
    parser.add_argument("--query-baseline-comparison", type=Path, help="Query baseline comparison JSON.")
    parser.add_argument("--daily-variant-metrics", type=Path, help="Daily-feed variant metrics JSON.")
    parser.add_argument("--diversity-sweep", type=Path, help="Diversity sweep metrics JSON.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/eval_figures"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    query_aggregate = load_json(args.query_aggregate) if args.query_aggregate else None
    daily_variant_metrics = load_json(args.daily_variant_metrics) if args.daily_variant_metrics else None

    if args.query_eval:
        scores_by_eval_set = compute_query_scores_from_eval_json(args.query_eval)
        computed_metrics = compute_metrics(scores_by_eval_set)
        print(f"Computed query metrics from {args.query_eval}: {len(scores_by_eval_set)} eval sets.")
        if query_aggregate is None:
            query_aggregate = computed_metrics

    rows = _headline_rows(query_aggregate, daily_variant_metrics)
    if rows:
        plot_headline_table(rows, args.out_dir)
    else:
        print("Skipping headline metrics table: no query or daily aggregate metrics supplied.")

    if query_aggregate:
        plot_query_case_metrics(query_aggregate, args.out_dir)
    else:
        print("Skipping query_search_case_metrics.png: --query-aggregate was not supplied.")

    if args.query_scores:
        plot_relevance_by_rank(load_json(args.query_scores), args.out_dir)
    else:
        print("Skipping query_relevance_by_rank outputs: --query-scores was not supplied.")

    if args.query_baseline_comparison:
        plot_baseline_comparison(load_json(args.query_baseline_comparison), args.out_dir)
    else:
        print("Skipping query baseline comparison plots: --query-baseline-comparison was not supplied.")

    if daily_variant_metrics:
        plot_daily_variant_comparison(daily_variant_metrics, args.out_dir)
    else:
        print("Skipping daily_feed_variant_comparison.png: --daily-variant-metrics was not supplied.")

    if args.diversity_sweep:
        plot_diversity_sweep(load_json(args.diversity_sweep), args.out_dir)
    else:
        print("Skipping diversity_index_sweep.png: --diversity-sweep was not supplied.")

    print(f"Wrote evaluation outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
