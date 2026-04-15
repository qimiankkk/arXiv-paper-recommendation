"""Google Scholar profile parser for user onboarding.

Fetches paper titles from a public Google Scholar profile URL,
then optionally enriches with abstracts via the Semantic Scholar API.
Returns a list of dicts compatible with EmbeddingModel.embed_papers().
"""

from __future__ import annotations

import time
import re
from html import unescape
from urllib.parse import urlparse, parse_qs

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

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"


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


def fetch_scholar_titles(user_id: str, max_papers: int = 30) -> list[str]:
    """Scrape paper titles from a Google Scholar profile.

    Args:
        user_id: The Google Scholar user ID.
        max_papers: Maximum titles to retrieve. Default 30.

    Returns:
        List of paper title strings.
    """
    url = (
        f"https://scholar.google.com/citations"
        f"?user={user_id}&cstart=0&pagesize={max_papers}&sortby=pubdate"
    )
    resp = requests.get(url, headers=SCHOLAR_HEADERS, timeout=15)
    resp.raise_for_status()

    if BeautifulSoup is not None:
        soup = BeautifulSoup(resp.text, "html.parser")
        title_links = soup.select("a.gsc_a_at")
        return [a.get_text(strip=True) for a in title_links[:max_papers]]

    matches = re.findall(r'<a[^>]*class="[^"]*gsc_a_at[^"]*"[^>]*>(.*?)</a>', resp.text)
    return [unescape(re.sub(r"<[^>]+>", "", m)).strip() for m in matches[:max_papers]]


def enrich_with_abstracts(
    titles: list[str],
    delay: float = 0.5,
) -> list[dict]:
    """Look up abstracts for paper titles via Semantic Scholar.

    Falls back to empty abstract if not found. Returns dicts with
    "title" and "abstract" keys, compatible with EmbeddingModel.embed_papers().

    Args:
        titles: List of paper title strings.
        delay: Seconds between API calls to respect rate limits.

    Returns:
        List of {"title": str, "abstract": str} dicts.
    """
    results = []
    for title in titles:
        abstract = ""
        try:
            resp = requests.get(
                S2_API,
                params={"query": title, "limit": 1, "fields": "title,abstract"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data and data[0].get("abstract"):
                    abstract = data[0]["abstract"]
        except requests.RequestException:
            pass

        results.append({"title": title, "abstract": abstract})
        time.sleep(delay)

    return results


def load_scholar_papers(
    scholar_url: str,
    max_papers: int = 30,
) -> list[dict] | None:
    """End-to-end: URL -> list of {"title", "abstract"} dicts.

    Args:
        scholar_url: A Google Scholar profile URL.
        max_papers: Maximum papers to retrieve.

    Returns:
        List of paper dicts, or None if the URL is invalid or scraping fails.
    """
    user_id = parse_scholar_url(scholar_url)
    if user_id is None:
        return None

    try:
        titles = fetch_scholar_titles(user_id, max_papers)
    except requests.RequestException:
        return None

    if not titles:
        return None

    return enrich_with_abstracts(titles)
