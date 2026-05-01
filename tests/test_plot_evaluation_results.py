from __future__ import annotations

import importlib.util
import json

import pytest

from scripts import plot_evaluation_results as plots


HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _daily_case(variant: str, delta: float, value_offset: float = 0.0):
    return {
        "user_profile": "profile",
        "inferred_threads": ["thread"],
        "k_u": 1,
        "algorithm_variant": variant,
        "algorithm_variant_label": plots.DAILY_FEED_VARIANT_LABELS[variant],
        "diversity_index_delta": delta,
        "computed_metrics": {
            "strict_precision@20": 0.4 + value_offset,
            "soft_precision@20": 0.7 + value_offset,
            "mean_relevance@20": 0.6 + value_offset,
            "ndcg@20": 0.8 + value_offset,
            "thread_coverage@20": 0.9,
            "redundancy@20": 0.3 + value_offset,
        },
        "papers": [
            {
                "rank": rank,
                "r_feed": 1.0 + value_offset,
                "user_interest_relevance": 1,
                "usefulness": 1,
                "freshness_or_timeliness": 1,
                "nearest_user_thread_id": 0,
                "nearest_user_thread_label": "thread",
                "is_relevant_for_coverage": True,
            }
            for rank in range(1, 21)
        ],
    }


def _daily_payload():
    evaluations = {}
    i = 1
    for delta in (0.0, 0.5):
        for variant_i, variant in enumerate(plots.DAILY_FEED_VARIANT_ORDER):
            evaluations[f"ev{i}"] = _daily_case(variant, delta, value_offset=variant_i * 0.05)
            i += 1
    return {"evaluations": evaluations}


def test_load_json_reports_path_and_unescaped_title_hint(tmp_path):
    bad_path = tmp_path / "bad.json"
    bad_path.write_text('{"title": "What Do Researchers Mean by "Reproducible"?"}', encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        plots.load_json(bad_path)

    message = str(exc.value)
    assert str(bad_path) in message
    assert "Invalid JSON" in message
    assert "Reproducible" in message


def test_compute_query_scores_from_eval_json_preserves_rank_order(tmp_path):
    eval_path = _write_json(
        tmp_path / "query_eval.json",
        {
            "evaluations": [
                {
                    "eval_set": "ev1",
                    "query_relevance": 2,
                    "user_interest_relevance": 1,
                    "usefulness": 0,
                },
                {
                    "eval_set": "ev1",
                    "query_relevance": 0,
                    "user_interest_relevance": 2,
                    "usefulness": 1,
                },
                {
                    "eval_set": "ev2",
                    "query_relevance": 1,
                    "user_interest_relevance": 1,
                    "usefulness": 1,
                },
            ]
        },
    )

    assert plots.compute_query_scores_from_eval_json(eval_path) == {
        "ev1": [1.5, 0.7],
        "ev2": [1.0],
    }


def test_compute_metrics_uses_thresholds_and_normalized_mean_relevance():
    metrics = plots.compute_metrics(
        {
            "ev2": [2.0, 1.0, 0.0],
            "ev1": [1.5, 1.4, 0.2],
        }
    )

    assert metrics["ev1"]["strict_precision@20"] == pytest.approx(1 / 3)
    assert metrics["ev1"]["soft_precision@20"] == pytest.approx(2 / 3)
    assert metrics["ev1"]["mean_relevance@20"] == pytest.approx((1.5 + 1.4 + 0.2) / 3 / 2)
    assert 0 <= metrics["ev1"]["ndcg@20"] <= 1
    assert metrics["overall_average"]["strict_precision@20"] == pytest.approx((1 / 3 + 1 / 3) / 2)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib is not installed")
def test_headline_table_outputs_csv_and_png(tmp_path):
    df = plots.plot_headline_table(
        [
            {
                "Algorithm": "Query Search",
                "strict_precision@20": 0.12345,
                "soft_precision@20": 0.98765,
                "mean_relevance@20": 0.55555,
                "ndcg@20": 0.99999,
            }
        ],
        tmp_path,
    )

    assert df.loc[0, "Strict P@20"] == 0.123
    assert (tmp_path / "headline_metrics_table.csv").exists()
    assert (tmp_path / "headline_metrics_table.png").exists()


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib is not installed")
def test_relevance_by_rank_outputs_csv_and_png(tmp_path):
    df = plots.plot_relevance_by_rank(
        {
            "ev1": {"p1": 2.0, "p2": 1.0},
            "ev2": {"p1": 1.0, "p2": 0.0},
        },
        tmp_path,
    )

    assert df.loc[0, "avg_score"] == 1.5
    assert df.loc[1, "avg_score"] == 0.5
    assert (tmp_path / "query_relevance_by_rank.csv").exists()
    assert (tmp_path / "query_relevance_by_rank.png").exists()


def test_main_skips_optional_plots(monkeypatch, tmp_path, capsys):
    aggregate_path = _write_json(
        tmp_path / "aggregate.json",
        {
            "overall_average": {
                "strict_precision@20": 0.4,
                "soft_precision@20": 0.8,
                "mean_relevance@20": 0.6,
                "ndcg@20": 0.9,
            },
            "ev1": {
                "strict_precision@20": 0.4,
                "soft_precision@20": 0.8,
                "mean_relevance@20": 0.6,
                "ndcg@20": 0.9,
            },
        },
    )
    calls = []
    monkeypatch.setattr(plots, "plot_headline_table", lambda rows, out_dir: calls.append("headline"))
    monkeypatch.setattr(plots, "plot_query_case_metrics", lambda metrics, out_dir: calls.append("cases"))

    assert plots.main(["--query-aggregate", str(aggregate_path), "--out-dir", str(tmp_path / "out")]) == 0

    captured = capsys.readouterr()
    assert calls == ["headline", "cases"]
    assert "Skipping query_relevance_by_rank outputs" in captured.out
    assert "Skipping query baseline comparison plots" in captured.out


def test_filter_daily_feed_cases_uses_default_delta(tmp_path):
    path = _write_json(tmp_path / "daily.json", _daily_payload())
    cases = plots.load_daily_feed_evaluation_cases(path)
    filtered = plots.filter_daily_feed_cases(cases, delta=0.5)

    assert len(cases) == 6
    assert len(filtered) == 3
    assert {case["diversity_index_delta"] for case in filtered} == {0.5}
    assert [case["algorithm_variant"] for case in filtered] == list(plots.DAILY_FEED_VARIANT_ORDER)


def test_daily_feed_rank_summary_normalizes_r_feed(tmp_path):
    path = _write_json(tmp_path / "daily.json", _daily_payload())
    cases = plots.filter_daily_feed_cases(plots.load_daily_feed_evaluation_cases(path), delta=0.5)
    summary = plots.summarize_daily_feed_relevance_by_rank(cases, delta_label="delta = 0.5")
    first = summary[
        (summary["algorithm_variant"] == "single_vector_average_baseline")
        & (summary["rank"] == 1)
    ].iloc[0]

    assert first["mean_normalized_r_feed"] == 0.5
    assert first["n"] == 1


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib is not installed")
def test_cli_smoke_writes_query_only_artifacts(tmp_path):
    aggregate_path = _write_json(
        tmp_path / "aggregate.json",
        {
            "overall_average": {
                "strict_precision@20": 0.4,
                "soft_precision@20": 0.8,
                "mean_relevance@20": 0.6,
                "ndcg@20": 0.9,
            },
            "ev1": {
                "strict_precision@20": 0.4,
                "soft_precision@20": 0.8,
                "mean_relevance@20": 0.6,
                "ndcg@20": 0.9,
            },
        },
    )
    out_dir = tmp_path / "figures"

    assert plots.main(["--query-aggregate", str(aggregate_path), "--out-dir", str(out_dir)]) == 0

    assert (out_dir / "headline_metrics_table.csv").exists()
    assert (out_dir / "headline_metrics_table.png").exists()
    assert (out_dir / "query_search_case_metrics.png").exists()


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib is not installed")
def test_daily_feed_plot_smoke_writes_expected_outputs(tmp_path):
    path = _write_json(tmp_path / "daily.json", _daily_payload())
    out_dir = tmp_path / "figures"

    result = plots.run_daily_feed_evaluation_plots(path, out_dir, delta=0.5)

    expected = {
        "daily_feed_relevance_metrics_by_variant.png",
        "daily_feed_relevance_metrics_by_variant.pdf",
        "daily_feed_relevance_metrics_by_variant.csv",
        "daily_feed_diversity_metrics_by_variant.png",
        "daily_feed_diversity_metrics_by_variant.pdf",
        "daily_feed_diversity_metrics_by_variant.csv",
        "daily_feed_relevance_by_rank_by_variant.png",
        "daily_feed_relevance_by_rank_by_variant.pdf",
        "daily_feed_relevance_by_rank_by_variant.csv",
    }
    assert result["loaded_cases"] == 6
    assert result["used_cases"] == 3
    assert {path.name for path in out_dir.iterdir()} == expected
