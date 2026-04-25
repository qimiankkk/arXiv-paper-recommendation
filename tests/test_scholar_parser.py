"""Tests for pipeline.scholar_parser module."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import requests

from pipeline.scholar_parser import (
    _extract_profile_name,
    _is_first_author,
    _normalize_last_name,
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
# _extract_profile_name
# ===================================================================

class TestExtractProfileName:

    def test_from_bs4_soup(self):
        from bs4 import BeautifulSoup as BS4
        soup = BS4(MOCK_SCHOLAR_HTML, "html.parser")
        assert _extract_profile_name(soup) == "Jane Smith"

    def test_from_raw_html(self):
        assert _extract_profile_name(MOCK_SCHOLAR_HTML) == "Jane Smith"

    def test_missing_element(self):
        assert _extract_profile_name("<html><body></body></html>") == ""


# ===================================================================
# _normalize_last_name
# ===================================================================

class TestNormalizeLastName:

    def test_simple_name(self):
        assert _normalize_last_name("John Smith") == "smith"

    def test_comma_format(self):
        assert _normalize_last_name("Smith, J.") == "smith"

    def test_single_name(self):
        assert _normalize_last_name("Smith") == "smith"

    def test_abbreviated_first(self):
        assert _normalize_last_name("J Smith") == "smith"

    def test_empty_string(self):
        assert _normalize_last_name("") == ""

    def test_hyphenated_last_name(self):
        assert _normalize_last_name("Marie Curie-Smith") == "curie-smith"


# ===================================================================
# _is_first_author
# ===================================================================

class TestIsFirstAuthor:

    def test_match(self):
        assert _is_first_author("Jane Smith", "J Smith, A Johnson") is True

    def test_no_match(self):
        assert _is_first_author("Jane Smith", "A Johnson, J Smith") is False

    def test_empty_authors(self):
        assert _is_first_author("Jane Smith", "") is False

    def test_empty_profile_name(self):
        assert _is_first_author("", "J Smith, A Johnson") is False

    def test_single_author(self):
        assert _is_first_author("Jane Smith", "J Smith") is True

    def test_case_insensitive(self):
        assert _is_first_author("JANE SMITH", "j smith, other") is True


# ===================================================================
# fetch_scholar_papers
# ===================================================================

class TestFetchScholarPapers:

    @patch("pipeline.scholar_parser.requests.get")
    def test_extracts_all_fields(self, mock_get):
        mock_get.return_value = _mock_response()

        papers, profile_name = fetch_scholar_papers("abc123")

        assert profile_name == "Jane Smith"
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

        papers, _ = fetch_scholar_papers("abc123")
        # Paper 4 ("Old Unpopular Paper") has empty citation text
        assert papers[3]["citations"] == 0

    @patch("pipeline.scholar_parser.requests.get")
    def test_max_papers_limit(self, mock_get):
        mock_get.return_value = _mock_response()

        papers, _ = fetch_scholar_papers("abc123", max_papers=2)
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

        papers, _ = fetch_scholar_papers("abc123")
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

    @patch("pipeline.scholar_parser.datetime")
    def test_citation_filter(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 1, 1)
        papers = [self._paper("High Cite", citations=50, year=2010)]
        result = filter_papers(papers, "Nobody")
        assert len(result) == 1
        assert result[0]["title"] == "High Cite"

    @patch("pipeline.scholar_parser.datetime")
    def test_recent_year_filter(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 1, 1)
        # cutoff_year = 2021, so year=2022 qualifies
        papers = [self._paper("Recent", citations=0, year=2022)]
        result = filter_papers(papers, "Nobody")
        assert len(result) == 1

    @patch("pipeline.scholar_parser.datetime")
    def test_year_exactly_at_cutoff(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 1, 1)
        # cutoff_year = 2021, year=2021 should pass (>=)
        papers = [self._paper("Cutoff", citations=0, year=2021)]
        result = filter_papers(papers, "Nobody")
        assert len(result) == 1

    @patch("pipeline.scholar_parser.datetime")
    def test_year_just_below_cutoff(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 1, 1)
        # cutoff_year = 2021, year=2020 should NOT pass via year condition
        papers = [self._paper("Old", citations=0, year=2020)]
        result = filter_papers(papers, "Nobody")
        # Falls back to returning all papers since nothing matched
        assert len(result) == 1

    @patch("pipeline.scholar_parser.datetime")
    def test_first_author_filter(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 1, 1)
        papers = [self._paper("FA", authors="J Smith, Other", citations=0, year=2010)]
        result = filter_papers(papers, "Jane Smith")
        assert len(result) == 1

    @patch("pipeline.scholar_parser.datetime")
    def test_no_condition_met_uses_fallback(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 1, 1)
        papers = [
            self._paper("A", citations=0, year=2010),
            self._paper("B", citations=5, year=2015),
        ]
        result = filter_papers(papers, "Nobody")
        # Fallback: return all (both fail all 3 conditions)
        assert len(result) == 2

    @patch("pipeline.scholar_parser.datetime")
    def test_max_n_enforced(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 1, 1)
        papers = [self._paper(f"P{i}", citations=100, year=2025) for i in range(30)]
        result = filter_papers(papers, "Nobody", max_n=10)
        assert len(result) == 10

    @patch("pipeline.scholar_parser.datetime")
    def test_sorted_by_citations_when_trimming(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 1, 1)
        papers = [
            self._paper("Low", citations=25, year=2025),
            self._paper("High", citations=500, year=2025),
            self._paper("Mid", citations=100, year=2025),
        ]
        result = filter_papers(papers, "Nobody", max_n=2)
        assert result[0]["title"] == "High"
        assert result[1]["title"] == "Mid"

    @patch("pipeline.scholar_parser.datetime")
    def test_empty_input(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 1, 1)
        assert filter_papers([], "Nobody") == []

    @patch("pipeline.scholar_parser.datetime")
    def test_mixed_conditions(self, mock_dt):
        """Papers matching different conditions are all included."""
        mock_dt.now.return_value = datetime(2026, 1, 1)
        papers = [
            self._paper("HighCite", citations=50, year=2010),           # citation
            self._paper("Recent", citations=0, year=2024),              # year
            self._paper("FirstAuth", authors="J Doe", citations=0, year=2010),  # first author
            self._paper("Nothing", citations=0, year=2010),             # none
        ]
        result = filter_papers(papers, "Jane Doe")
        titles = {p["title"] for p in result}
        assert "HighCite" in titles
        assert "Recent" in titles
        assert "FirstAuth" in titles
        # "Nothing" only appears via fallback, but since 3 papers passed, no fallback
        assert "Nothing" not in titles


# ===================================================================
# load_scholar_papers (end-to-end with mocks)
# ===================================================================

class TestLoadScholarPapersE2E:

    @patch("pipeline.scholar_parser.datetime")
    @patch("pipeline.scholar_parser.requests.get")
    def test_happy_path(self, mock_get, mock_dt):
        mock_dt.now.return_value = datetime(2026, 1, 1)
        mock_get.return_value = _mock_response()

        result = load_scholar_papers(
            "https://scholar.google.com/citations?user=abc123"
        )
        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0
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

    @patch("pipeline.scholar_parser.datetime")
    @patch("pipeline.scholar_parser.requests.get")
    def test_only_one_http_request(self, mock_get, mock_dt):
        """No Semantic Scholar calls — exactly 1 HTTP request."""
        mock_dt.now.return_value = datetime(2026, 1, 1)
        mock_get.return_value = _mock_response()

        load_scholar_papers(
            "https://scholar.google.com/citations?user=abc123"
        )
        assert mock_get.call_count == 1
        # Ensure the single call was to Scholar, not Semantic Scholar
        call_url = mock_get.call_args[0][0]
        assert "scholar.google.com" in call_url
