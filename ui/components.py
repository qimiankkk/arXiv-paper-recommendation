"""Reusable Streamlit UI widgets for the ArXiv Daily app.

Provides self-contained components that can be composed into pages:
- paper_card: renders a single paper with action buttons.
- topic_selector: multiselect for arXiv categories with human-readable labels.
- loading_spinner_with_message: context manager for custom spinner messages.
"""

from __future__ import annotations

from typing import Callable

import streamlit as st


# Human-readable label -> arXiv category code mapping.
# Only topics that exist in the corpus (category_centroids) will be shown.
TOPIC_LABELS: dict[str, str] = {
    "Machine Learning": "cs.LG",
    "Computer Vision": "cs.CV",
    "Natural Language Processing": "cs.CL",
    "Robotics": "cs.RO",
    "Statistics / ML Theory": "stat.ML",
    "Artificial Intelligence": "cs.AI",
    "Computation & Language": "cs.CL",
    "Neural Networks": "cs.NE",
    "Information Retrieval": "cs.IR",
    "Human-Computer Interaction": "cs.HC",
    "Cryptography & Security": "cs.CR",
    "Distributed Computing": "cs.DC",
    "Computational Biology": "q-bio.QM",
    "Physics & ML": "physics.comp-ph",
    "Quantitative Finance": "q-fin.CP",
}


def paper_card(
    meta: dict,
    on_like: Callable[[str], None],
    on_save: Callable[[str], None],
    on_skip: Callable[[str], None],
) -> None:
    """Render a single paper as a Streamlit card with action buttons.

    Args:
        meta: Paper metadata dict with keys: id, title, abstract, categories,
            update_date, cluster_id, rec_score.
        on_like: Callback invoked with arxiv_id when Like is clicked.
        on_save: Callback invoked with arxiv_id when Save is clicked.
        on_skip: Callback invoked with arxiv_id when Skip is clicked.
    """
    arxiv_id = meta["id"]
    responded = st.session_state.get("responded", set())
    is_responded = arxiv_id in responded

    with st.container(border=True):
        st.subheader(meta["title"])

        # Category badges
        categories = meta.get("categories", [])
        if categories:
            st.caption(" ".join(f"`{cat}`" for cat in categories))

        # Abstract snippet
        abstract = meta.get("abstract", "")
        if len(abstract) > 300:
            abstract = abstract[:300] + "..."
        st.write(abstract)

        # Links
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        st.markdown(f"[ArXiv page]({abs_url}) | [PDF]({pdf_url})")

        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(
                "Like" if not is_responded else "Liked",
                key=f"like_{arxiv_id}",
                disabled=is_responded,
            ):
                on_like(arxiv_id)
        with col2:
            if st.button(
                "Save" if not is_responded else "Saved",
                key=f"save_{arxiv_id}",
                disabled=is_responded,
            ):
                on_save(arxiv_id)
        with col3:
            if st.button(
                "Skip" if not is_responded else "Skipped",
                key=f"skip_{arxiv_id}",
                disabled=is_responded,
            ):
                on_skip(arxiv_id)


def topic_selector(category_centroids: dict) -> list[str]:
    """Render a multiselect widget for arXiv topic categories.

    Only shows topics that exist in the loaded corpus.

    Args:
        category_centroids: Dict mapping arXiv category string to centroid vector.

    Returns:
        List of selected arXiv category code strings, e.g. ["cs.LG", "cs.CV"].
    """
    # Filter to topics that exist in the corpus
    available: dict[str, str] = {
        label: code
        for label, code in TOPIC_LABELS.items()
        if code in category_centroids
    }

    selected_labels = st.multiselect(
        "Select your research interests",
        options=list(available.keys()),
        default=None,
    )

    # Map labels back to category codes (deduplicate)
    selected_codes = list({available[label] for label in selected_labels})
    return selected_codes


def loading_spinner_with_message(message: str):
    """Context manager wrapping st.spinner with a custom message.

    Args:
        message: The message to display inside the spinner.

    Returns:
        A context manager (st.spinner instance).
    """
    return st.spinner(message)
