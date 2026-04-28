"""Google Scholar profile parser for user onboarding.

Fetches paper metadata from a public Google Scholar profile URL in a
single HTTP request.  Extracts title, authors, venue, citation count,
and publication year for each paper, then filters to the most
representative subset for embedding.

Returns a list of dicts compatible with EmbeddingModel.embed_papers().
"""

from __future__ import annotations

import re
from datetime import datetime
from html import unescape
from urllib.parse import urlparse, parse_qs, urljoin

import requests

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:
    BeautifulSoup = None


SCHOLAR_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def parse_scholar_url(url: str) -> str | None:
    """Extract the Google Scholar user ID from a profile URL.

    Args:
        url: A Google Scholar profile URL.

    Returns:
        The user ID string, or None if the URL is not a valid profile.
    """
    parsed = urlparse(url)
    if not parsed.hostname or "scholar.google" not in parsed.hostname:
        return None
    params = parse_qs(parsed.query)
    user_ids = params.get("user")
    return user_ids[0] if user_ids else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_profile_name(soup_or_html) -> str:
    """Extract the profile holder's name from the Scholar page.

    Accepts either a BeautifulSoup object or a raw HTML string.
    Returns "" if the name element is not found.
    """
    if BeautifulSoup is not None and isinstance(soup_or_html, BeautifulSoup):
        el = soup_or_html.select_one("#gsc_prf_in")
        return el.get_text(strip=True) if el else ""

    # Regex fallback for raw HTML string
    m = re.search(r'<div[^>]*id="gsc_prf_in"[^>]*>(.*?)</div>', str(soup_or_html))
    if m:
        return unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()
    return ""


def _parse_int(text: str) -> int:
    """Parse an integer from text, returning 0 for empty or non-numeric."""
    digits = re.sub(r"\D", "", text)
    return int(digits) if digits else 0


def _normalize_last_name(name: str) -> str:
    """Extract a normalized last name for fuzzy author matching.

    Handles: "John Smith", "Smith, J.", "J Smith", "Smith".
    Returns lowercase last name stripped of trailing punctuation.
    """
    name = name.strip()
    if not name:
        return ""
    if "," in name:
        last = name.split(",")[0].strip()
    else:
        last = name.split()[-1]
    return last.lower().rstrip(".")


def _is_first_author(profile_name: str, authors_str: str) -> bool:
    """Check if the profile holder is the first author.

    Compares last names only, case-insensitive.
    """
    if not profile_name or not authors_str:
        return False
    first_author = authors_str.split(",")[0].strip()
    return _normalize_last_name(profile_name) == _normalize_last_name(first_author)


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def _parse_rows_bs4(soup, max_papers: int) -> list[dict]:
    """Extract paper metadata from Scholar HTML using BeautifulSoup."""
    papers: list[dict] = []
    for row in soup.select("tr.gsc_a_tr")[:max_papers]:
        title_el = row.select_one("a.gsc_a_at")
        if not title_el:
            continue
        title = title_el.get_text(strip=True)
        href = title_el.get("href", "")
        paper_url = urljoin("https://scholar.google.com", href) if href else ""

        gray_divs = row.select("td.gsc_a_t div.gs_gray")
        authors = gray_divs[0].get_text(strip=True) if len(gray_divs) > 0 else ""
        venue = gray_divs[1].get_text(strip=True) if len(gray_divs) > 1 else ""

        cite_el = row.select_one("td.gsc_a_c")
        citations = _parse_int(cite_el.get_text(strip=True)) if cite_el else 0

        year_el = row.select_one("td.gsc_a_y span.gsc_a_h")
        year = _parse_int(year_el.get_text(strip=True)) if year_el else 0

        papers.append({
            "title": title,
            "authors": authors,
            "venue": venue,
            "citations": citations,
            "year": year,
            "abstract": "",
            "url": paper_url,
        })
    return papers


def _parse_rows_regex(html: str, max_papers: int) -> list[dict]:
    """Extract paper metadata from Scholar HTML using regex (fallback)."""
    papers: list[dict] = []
    row_matches = re.findall(
        r'<tr[^>]*class="[^"]*gsc_a_tr[^"]*"[^>]*>(.*?)</tr>',
        html,
        re.DOTALL,
    )
    for row_html in row_matches[:max_papers]:
        title_m = re.search(
            r'<a[^>]*class="[^"]*gsc_a_at[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            row_html,
        )
        if not title_m:
            continue
        paper_url = urljoin("https://scholar.google.com", title_m.group(1))
        title = unescape(re.sub(r"<[^>]+>", "", title_m.group(2))).strip()

        gray = re.findall(
            r'<div[^>]*class="[^"]*gs_gray[^"]*"[^>]*>(.*?)</div>', row_html
        )
        authors = unescape(re.sub(r"<[^>]+>", "", gray[0])).strip() if len(gray) > 0 else ""
        venue = unescape(re.sub(r"<[^>]+>", "", gray[1])).strip() if len(gray) > 1 else ""

        cite_m = re.search(
            r'<td[^>]*class="[^"]*gsc_a_c[^"]*"[^>]*>(.*?)</td>', row_html, re.DOTALL
        )
        citations = _parse_int(cite_m.group(1)) if cite_m else 0

        year_m = re.search(
            r'<td[^>]*class="[^"]*gsc_a_y[^"]*"[^>]*>.*?(\d{4}).*?</td>',
            row_html,
            re.DOTALL,
        )
        year = int(year_m.group(1)) if year_m else 0

        papers.append({
            "title": title,
            "authors": authors,
            "venue": venue,
            "citations": citations,
            "year": year,
            "abstract": "",
            "url": paper_url,
        })
    return papers


def _extract_description_bs4(soup) -> str:
    """Extract paper description/abstract from a Scholar citation page."""
    for row in soup.select("div.gsc_oci_row"):
        field_el = row.select_one("div.gsc_oci_field")
        value_el = row.select_one("div.gsc_oci_value")
        if not field_el or not value_el:
            continue
        field = field_el.get_text(" ", strip=True).lower()
        if field in {"description", "abstract", "summary"}:
            text = value_el.get_text(" ", strip=True)
            if text:
                return text

    descr_el = soup.select_one("#gsc_oci_descr")
    if descr_el:
        text = descr_el.get_text(" ", strip=True)
        if text:
            return text

    meta_el = soup.select_one('meta[name="description"]')
    if meta_el and meta_el.get("content"):
        return meta_el["content"].strip()
    return ""


def _extract_description_regex(html: str) -> str:
    """Regex fallback to extract description from citation detail page."""
    rows = re.findall(
        r'<div[^>]*class="[^"]*gsc_oci_row[^"]*"[^>]*>(.*?)</div>\s*</div>',
        html,
        re.DOTALL,
    )
    for row in rows:
        field_m = re.search(r'gsc_oci_field[^>]*>(.*?)</div>', row, re.DOTALL)
        value_m = re.search(r'gsc_oci_value[^>]*>(.*?)</div>', row, re.DOTALL)
        if not field_m or not value_m:
            continue
        field = unescape(re.sub(r"<[^>]+>", "", field_m.group(1))).strip().lower()
        if field in {"description", "abstract", "summary"}:
            text = unescape(re.sub(r"<[^>]+>", "", value_m.group(1))).strip()
            if text:
                return text

    descr_m = re.search(r'id="gsc_oci_descr"[^>]*>(.*?)</div>', html, re.DOTALL)
    if descr_m:
        text = unescape(re.sub(r"<[^>]+>", "", descr_m.group(1))).strip()
        if text:
            return text

    meta_m = re.search(
        r'<meta[^>]*name="description"[^>]*content="([^"]+)"[^>]*>',
        html,
    )
    if meta_m:
        return unescape(meta_m.group(1)).strip()
    return ""


def fetch_scholar_paper_description(paper_url: str) -> str:
    """Fetch description text from a Scholar citation detail URL."""
    if not paper_url:
        return ""
    resp = requests.get(paper_url, headers=SCHOLAR_HEADERS, timeout=15)
    resp.raise_for_status()

    if BeautifulSoup is not None:
        soup = BeautifulSoup(resp.text, "html.parser")
        return _extract_description_bs4(soup)
    return _extract_description_regex(resp.text)


def enrich_scholar_papers_with_descriptions(papers: list[dict]) -> list[dict]:
    """Populate each paper['abstract'] from its Scholar detail page."""
    enriched: list[dict] = []
    for paper in papers:
        p = dict(paper)
        paper_url = p.get("url", "")
        try:
            description = fetch_scholar_paper_description(paper_url)
        except requests.RequestException:
            description = ""
        p["abstract"] = description or p.get("abstract", "") or ""
        enriched.append(p)
    return enriched


def fetch_scholar_papers(
    user_id: str,
    max_papers: int = 30,
    sort_by_pubdate: bool = True,
) -> tuple[list[dict], str]:
    """Scrape paper metadata and profile name from a Google Scholar profile.

    Makes exactly one HTTP request.

    Args:
        user_id: The Google Scholar user ID.
        max_papers: Maximum papers to retrieve. Default 30.
        sort_by_pubdate: If True, request newest-first ordering from Scholar.
            If False, use Scholar's default ordering (typically citation-based).

    Returns:
        Tuple of (papers, profile_name).  Each paper dict has keys:
        title, authors, venue, citations (int), year (int), abstract ("").
    """
    url = (
        f"https://scholar.google.com/citations"
        f"?user={user_id}&cstart=0&pagesize={max_papers}"
    )
    if sort_by_pubdate:
        url += "&sortby=pubdate"
    resp = requests.get(url, headers=SCHOLAR_HEADERS, timeout=15)
    resp.raise_for_status()

    if BeautifulSoup is not None:
        soup = BeautifulSoup(resp.text, "html.parser")
        profile_name = _extract_profile_name(soup)
        papers = _parse_rows_bs4(soup, max_papers)
    else:
        profile_name = _extract_profile_name(resp.text)
        papers = _parse_rows_regex(resp.text, max_papers)

    return papers, profile_name


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_papers(
    papers: list[dict],
    profile_name: str,
    max_n: int = 15,
) -> list[dict]:
    """Select the most representative papers for embedding.

    A paper is kept if it satisfies ANY of:
    - More than 20 citations
    - Published in the recent 5 years
    - Profile holder is the first author

    If more than *max_n* papers pass, the top *max_n* by citation count
    are kept.  If zero papers pass, falls back to ``papers[:max_n]``.

    Args:
        papers: Paper dicts from :func:`fetch_scholar_papers`.
        profile_name: Profile holder's name for first-author detection.
        max_n: Maximum papers to return.  Default 15.

    Returns:
        Filtered list of paper dicts, length <= *max_n*.
    """
    if not papers:
        return []

    current_year = datetime.now().year
    cutoff_year = current_year - 5

    passed: list[dict] = []
    for p in papers:
        if p["citations"] > 20:
            passed.append(p)
        elif p["year"] > 0 and p["year"] >= cutoff_year:
            passed.append(p)
        elif _is_first_author(profile_name, p["authors"]):
            passed.append(p)

    # Fallback: if no papers matched any condition, return what we have
    if not passed:
        passed = list(papers)

    # Sort by citations descending and cap at max_n
    passed.sort(key=lambda p: p["citations"], reverse=True)
    return passed[:max_n]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_scholar_papers(
    scholar_url: str,
    max_papers: int = 30,
) -> list[dict] | None:
    """End-to-end: URL -> filtered list of paper dicts.

    Args:
        scholar_url: A Google Scholar profile URL.
        max_papers: Maximum papers to scrape from the page.

    Returns:
        List of paper dicts (filtered to ~15), or None if the URL is
        invalid or scraping fails.  Each dict has at least "title" and
        "abstract" keys for compatibility with
        :meth:`EmbeddingModel.embed_papers`.
    """
    user_id = parse_scholar_url(scholar_url)
    if user_id is None:
        return None

    try:
        papers, profile_name = fetch_scholar_papers(user_id, max_papers)
    except requests.RequestException:
        return None

    if not papers:
        return None

    return filter_papers(papers, profile_name)
