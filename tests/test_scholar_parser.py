"""
Tests for pipeline.scholar_parser module.

To run just the live tests: 
    pytest tests/test_scholar_parser.py -m live -v -s

To skip them: 
    pytest tests/test_scholar_parser.py -m "not live"
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from pipeline.scholar_parser import (
    fetch_scholar_papers,
    filter_papers,
    load_scholar_papers,
    parse_scholar_url,
)

# ---------------------------------------------------------------------------
# Fixtures: realistic Google Scholar profile HTML
# ---------------------------------------------------------------------------

MOCK_SCHOLAR_HTML = """\
<html>
<head><title>Jane Smith - Google Scholar</title></head>
<body>
<div id="gsc_prf_in">Jane Smith</div>
<table id="gsc_a_t">
<tbody id="gsc_a_b">

<tr class="gsc_a_tr">
  <td class="gsc_a_t">
    <a class="gsc_a_at" href="/citations?view_op=view_citation&amp;user=abc123">Deep Learning for NLP</a>
    <div class="gs_gray">J Smith, A Johnson, B Lee</div>
    <div class="gs_gray">Nature, 2023</div>
  </td>
  <td class="gsc_a_c"><a class="gsc_a_ac gs_ibl" href="#">150</a></td>
  <td class="gsc_a_y"><span class="gsc_a_h gsc_a_hc gs_ibl">2023</span></td>
</tr>

<tr class="gsc_a_tr">
  <td class="gsc_a_t">
    <a class="gsc_a_at" href="#">Attention Mechanisms Survey</a>
    <div class="gs_gray">A Johnson, J Smith</div>
    <div class="gs_gray">ICML, 2019</div>
  </td>
  <td class="gsc_a_c"><a class="gsc_a_ac gs_ibl" href="#">5</a></td>
  <td class="gsc_a_y"><span class="gsc_a_h gsc_a_hc gs_ibl">2019</span></td>
</tr>

<tr class="gsc_a_tr">
  <td class="gsc_a_t">
    <a class="gsc_a_at" href="#">Transformer Optimization</a>
    <div class="gs_gray">J Smith</div>
    <div class="gs_gray">arXiv preprint, 2024</div>
  </td>
  <td class="gsc_a_c"><a class="gsc_a_ac gs_ibl" href="#">3</a></td>
  <td class="gsc_a_y"><span class="gsc_a_h gsc_a_hc gs_ibl">2024</span></td>
</tr>

<tr class="gsc_a_tr">
  <td class="gsc_a_t">
    <a class="gsc_a_at" href="#">Old Unpopular Paper</a>
    <div class="gs_gray">B Lee, J Smith</div>
    <div class="gs_gray">Workshop on XYZ, 2010</div>
  </td>
  <td class="gsc_a_c"><a class="gsc_a_ac gs_ibl" href="#"></a></td>
  <td class="gsc_a_y"><span class="gsc_a_h gsc_a_hc gs_ibl">2010</span></td>
</tr>

</tbody>
</table>
</body>
</html>
"""


def _mock_response(html: str = MOCK_SCHOLAR_HTML) -> MagicMock:
    """Create a mock requests.Response with the given HTML."""
    resp = MagicMock()
    resp.text = html
    resp.raise_for_status = MagicMock()
    return resp


# ===================================================================
# parse_scholar_url
# ===================================================================

class TestParseScholarUrl:

    def test_valid_url(self):
        url = "https://scholar.google.com/citations?user=abc123&hl=en"
        assert parse_scholar_url(url) == "abc123"

    def test_valid_url_different_tld(self):
        url = "https://scholar.google.co.uk/citations?user=xyz789"
        assert parse_scholar_url(url) == "xyz789"

    def test_missing_user_param(self):
        url = "https://scholar.google.com/citations?hl=en"
        assert parse_scholar_url(url) is None

    def test_wrong_domain(self):
        url = "https://www.google.com/search?user=abc123"
        assert parse_scholar_url(url) is None

    def test_empty_string(self):
        assert parse_scholar_url("") is None

    def test_not_a_url(self):
        assert parse_scholar_url("not a url at all") is None

    def test_user_with_special_chars(self):
        url = "https://scholar.google.com/citations?user=Ab_C-12"
        assert parse_scholar_url(url) == "Ab_C-12"


# ===================================================================
# fetch_scholar_papers
# ===================================================================

class TestFetchScholarPapers:

    @patch("pipeline.scholar_parser.requests.get")
    def test_extracts_all_fields(self, mock_get):
        mock_get.return_value = _mock_response()

        papers = fetch_scholar_papers("abc123")
        assert len(papers) == 4

        p = papers[0]
        assert p["title"] == "Deep Learning for NLP"
        assert p["authors"] == "J Smith, A Johnson, B Lee"
        assert p["venue"] == "Nature, 2023"
        assert p["citations"] == 150
        assert p["year"] == 2023
        assert p["abstract"] == ""

    @patch("pipeline.scholar_parser.requests.get")
    def test_empty_citations_parsed_as_zero(self, mock_get):
        mock_get.return_value = _mock_response()

        papers = fetch_scholar_papers("abc123")
        # Paper 4 ("Old Unpopular Paper") has empty citation text
        assert papers[3]["citations"] == 0

    @patch("pipeline.scholar_parser.requests.get")
    def test_max_papers_limit(self, mock_get):
        mock_get.return_value = _mock_response()

        papers = fetch_scholar_papers("abc123", max_papers=2)
        assert len(papers) == 2

    @patch("pipeline.scholar_parser.requests.get")
    def test_request_url_format(self, mock_get):
        mock_get.return_value = _mock_response()

        fetch_scholar_papers("testuser", max_papers=20)
        call_url = mock_get.call_args[0][0]
        assert "user=testuser" in call_url
        assert "pagesize=20" in call_url
        assert "sortby=pubdate" in call_url

    @patch("pipeline.scholar_parser.requests.get")
    def test_abstract_always_empty(self, mock_get):
        mock_get.return_value = _mock_response()

        papers = fetch_scholar_papers("abc123")
        assert all(p["abstract"] == "" for p in papers)

    @patch("pipeline.scholar_parser.requests.get")
    def test_uses_scholar_headers(self, mock_get):
        mock_get.return_value = _mock_response()

        fetch_scholar_papers("abc123")
        _, kwargs = mock_get.call_args
        assert "User-Agent" in kwargs["headers"]


# ===================================================================
# filter_papers
# ===================================================================

class TestFilterPapers:

    @staticmethod
    def _paper(title="Test", authors="X Test", citations=0, year=2000):
        return {
            "title": title,
            "authors": authors,
            "venue": "",
            "citations": citations,
            "year": year,
            "abstract": "",
        }

    def test_selects_top_cited_and_top_recent(self):
        papers = [
            self._paper("High Cite 1", citations=100, year=2010),
            self._paper("High Cite 2", citations=80, year=2011),
            self._paper("High Cite 3", citations=60, year=2012),
            self._paper("Recent 1", citations=10, year=2026),
            self._paper("Recent 2", citations=5, year=2025),
            self._paper("Recent 3", citations=1, year=2024),
            self._paper("Other", citations=0, year=2000),
        ]
        result = filter_papers(papers)

        assert [p["title"] for p in result] == [
            "High Cite 1",
            "High Cite 2",
            "High Cite 3",
            "Recent 1",
            "Recent 2",
            "Recent 3",
        ]

    def test_deduplicates_overlap_between_rankings(self):
        papers = [
            self._paper("Paper A", citations=100, year=2026),
            self._paper("Paper B", citations=80, year=2025),
            self._paper("Paper C", citations=60, year=2024),
            self._paper("Paper D", citations=1, year=2023),
        ]
        result = filter_papers(papers)

        assert [p["title"] for p in result] == ["Paper A", "Paper B", "Paper C"]

    def test_deduplicates_titles_case_insensitively(self):
        papers = [
            self._paper("Paper A", citations=100, year=2020),
            self._paper("Paper B", citations=80, year=2019),
            self._paper("Paper C", citations=60, year=2018),
            self._paper(" paper   a ", citations=0, year=2026),
            self._paper("Paper D", citations=0, year=2025),
            self._paper("Paper E", citations=0, year=2024),
        ]
        result = filter_papers(papers)

        assert [p["title"] for p in result] == [
            "Paper A",
            "Paper B",
            "Paper C",
            "Paper D",
            "Paper E",
        ]

    def test_max_n_applies_to_each_ranking(self):
        papers = [
            self._paper("Cited 1", citations=100, year=2010),
            self._paper("Cited 2", citations=80, year=2011),
            self._paper("Cited 3", citations=60, year=2012),
            self._paper("Recent 1", citations=10, year=2026),
            self._paper("Recent 2", citations=5, year=2025),
            self._paper("Recent 3", citations=1, year=2024),
        ]
        result = filter_papers(papers, max_n=2)

        assert [p["title"] for p in result] == [
            "Cited 1",
            "Cited 2",
            "Recent 1",
            "Recent 2",
        ]

    def test_ties_keep_original_order(self):
        papers = [
            self._paper("A", citations=10, year=2020),
            self._paper("B", citations=10, year=2020),
            self._paper("C", citations=10, year=2020),
            self._paper("D", citations=0, year=2019),
        ]
        result = filter_papers(papers)

        assert [p["title"] for p in result] == ["A", "B", "C"]

    def test_first_author_status_does_not_affect_selection(self):
        papers = [
            self._paper("First Author", authors="J Smith", citations=0, year=2000),
            self._paper("Highly Cited", authors="Other", citations=100, year=1999),
            self._paper("Most Recent", authors="Other", citations=0, year=2026),
        ]
        result = filter_papers(papers, max_n=1)

        assert [p["title"] for p in result] == ["Highly Cited", "Most Recent"]

    def test_empty_input(self):
        assert filter_papers([]) == []

    def test_zero_max_n_returns_empty_list(self):
        assert filter_papers([self._paper("A", citations=1, year=2026)], max_n=0) == []


# ===================================================================
# load_scholar_papers (end-to-end with mocks)
# ===================================================================

class TestLoadScholarPapersE2E:

    @patch("pipeline.scholar_parser.requests.get")
    def test_happy_path(self, mock_get):
        mock_get.return_value = _mock_response()

        result = load_scholar_papers(
            "https://scholar.google.com/citations?user=abc123"
        )
        assert result is not None
        assert isinstance(result, list)
        assert [p["title"] for p in result] == [
            "Deep Learning for NLP",
            "Attention Mechanisms Survey",
            "Transformer Optimization",
        ]
        assert all("title" in p and "abstract" in p for p in result)

    def test_invalid_url_returns_none(self):
        assert load_scholar_papers("https://notscholar.com/fake") is None

    @patch("pipeline.scholar_parser.requests.get")
    def test_network_error_returns_none(self, mock_get):
        mock_get.side_effect = requests.RequestException("Connection failed")
        result = load_scholar_papers(
            "https://scholar.google.com/citations?user=abc123"
        )
        assert result is None

    @patch("pipeline.scholar_parser.requests.get")
    def test_empty_page_returns_none(self, mock_get):
        html = '<html><body><div id="gsc_prf_in">Test</div></body></html>'
        mock_get.return_value = _mock_response(html)

        result = load_scholar_papers(
            "https://scholar.google.com/citations?user=abc123"
        )
        assert result is None

    @patch("pipeline.scholar_parser.requests.get")
    def test_only_one_http_request(self, mock_get):
        """No Semantic Scholar calls; exactly 1 HTTP request."""
        mock_get.return_value = _mock_response()

        load_scholar_papers(
            "https://scholar.google.com/citations?user=abc123"
        )
        assert mock_get.call_count == 1
        # Ensure the single call was to Scholar, not Semantic Scholar
        call_url = mock_get.call_args[0][0]
        assert "scholar.google.com" in call_url


# ===================================================================
# Live integration tests against real Scholar profiles
# ===================================================================

import time as _time

LIVE_PROFILES = [
    ("https://scholar.google.com/citations?user=2w-C5G0AAAAJ&hl=en", "2w-C5G0AAAAJ"),
    ("https://scholar.google.com/citations?user=0RAmmIAAAAAJ&hl=en", "0RAmmIAAAAAJ"),
    ("https://scholar.google.com/citations?user=4Vb5nvIAAAAJ&hl=en", "4Vb5nvIAAAAJ"),
    ("https://scholar.google.com/citations?user=vx68rkMAAAAJ&hl=en", "vx68rkMAAAAJ"),
]


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_SCHOLAR") != "1",
    reason="Set RUN_LIVE_SCHOLAR=1 to run live Google Scholar tests.",
)
class TestLiveScholarProfiles:
    """Integration tests against real Google Scholar profiles.

    Run with:  pytest tests/test_scholar_parser.py -m live -v -s
    (The -s flag is required to see printed output.)
    """

    @pytest.mark.parametrize("url,user_id", LIVE_PROFILES)
    def test_full_pipeline(self, url, user_id):
        """Scrape, filter, and report diagnostics for a real profile."""

        # --- Step 1: Fetch ---
        t0 = _time.perf_counter()
        papers_raw = fetch_scholar_papers(user_id)
        t_fetch = _time.perf_counter() - t0

        print(f"\n{'='*70}")
        print(f"Profile user: {user_id}")
        print(f"URL:     {url}")
        print(f"Fetch time: {t_fetch:.2f}s")
        print(f"Papers scraped: {len(papers_raw)}")
        print(f"{'-'*70}")
        print(f"{'#':<4} {'Year':<6} {'Cites':<7} {'Authors (first 50)':<52} Title")
        print(f"{'-'*70}")
        for i, p in enumerate(papers_raw, 1):
            authors_short = p["authors"][:50] + ("..." if len(p["authors"]) > 50 else "")
            print(f"{i:<4} {p['year']:<6} {p['citations']:<7} {authors_short:<52} {p['title'][:]}")

        # --- Step 2: Filter ---
        t1 = _time.perf_counter()
        papers_filtered = filter_papers(papers_raw)
        t_filter = _time.perf_counter() - t1

        print(f"\n{'='*70}")
        print(f"FILTERED: {len(papers_filtered)} / {len(papers_raw)} papers  "
              f"(filter time: {t_filter*1000:.1f}ms)")
        print("Filter criteria: top 3 by citations plus top 3 by year")
        print(f"{'-'*70}")
        print(f"{'#':<4} {'Year':<6} {'Cites':<7} Title")
        print(f"{'-'*70}")
        for i, p in enumerate(papers_filtered, 1):
            print(f"{i:<4} {p['year']:<6} {p['citations']:<7} {p['title'][:60]}")

        # --- Step 3: End-to-end via load_scholar_papers ---
        t2 = _time.perf_counter()
        result = load_scholar_papers(url)
        t_total = _time.perf_counter() - t2

        print(f"\n{'='*70}")
        print(f"TOTAL load_scholar_papers() time: {t_total:.2f}s")
        print(f"{'='*70}\n")

        # --- Assertions ---
        assert papers_raw, "Should scrape at least 1 paper"
        assert papers_filtered, "Should have at least 1 filtered paper"
        assert len(papers_filtered) <= 6, f"Should cap at 6, got {len(papers_filtered)}"
        assert result is not None, "load_scholar_papers should succeed"
        assert all("title" in p and "abstract" in p for p in result), (
            "Every paper must have 'title' and 'abstract' keys"
        )

        # Verify all fields are populated
        for p in papers_raw:
            assert isinstance(p["title"], str) and p["title"]
            assert isinstance(p["citations"], int) and p["citations"] >= 0
            assert isinstance(p["year"], int)
            assert isinstance(p["authors"], str)
