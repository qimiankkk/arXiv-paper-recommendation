"""Onboarding page: three exclusive user-initialization modes."""

from __future__ import annotations

import streamlit as st

from pipeline.index import PaperIndex
from pipeline.embed import EmbeddingModel
from pipeline.scholar_parser import (
    enrich_scholar_papers_with_descriptions,
    fetch_scholar_papers,
    parse_scholar_url,
)
from user.db import create_user
from user.profile import (
    init_user_profile_from_description,
    init_user_profile_from_topics,
    init_user_profile_from_scholar_top3,
)
from ui.components import topic_selector


@st.cache_resource
def _get_embed_model() -> EmbeddingModel:
    return EmbeddingModel()


def render_onboarding(index: PaperIndex, db_path: str) -> None:
    st.title("ArXiv Daily")
    st.write("Personalized paper recommendations from arXiv, delivered daily.")
    st.divider()

    name = st.text_input("Your name", placeholder="Enter your display name")

    st.write("**Choose one initialization method (three options, no mixing):**")
    init_mode = st.radio(
        "Initialization mode",
        options=[
            "Natural language description",
            "Category tags",
            "Google Scholar profile",
        ],
        index=2,
    )

    selected_categories: list[str] = []
    user_description = ""
    scholar_url = ""

    if init_mode == "Natural language description":
        user_description = st.text_area(
            "Describe what papers you like",
            placeholder="Example: I like recent NLP papers on LLM alignment, retrieval-augmented generation, and multilingual evaluation.",
            height=120,
        )
    elif init_mode == "Category tags":
        st.write("**Pick topics you're interested in:**")
        selected_categories = topic_selector(index.category_centroids)
    else:
        scholar_url = st.text_input(
            "Google Scholar URL",
            placeholder="https://scholar.google.com/citations?user=...",
        )
        st.caption(
            "We use top 3 most-cited papers from this profile, then map each paper "
            "to its nearest arXiv category centroid."
        )

    # -- Diversity slider --
    st.write("**How broad should your daily papers be?**")
    diversity = st.slider(
        "Diversity",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0 = focused on your strongest interest · 1 = explore broadly",
    )

    if st.button("Start reading", type="primary"):
        if not name.strip():
            st.error("Please enter your name.")
            return

        model = _get_embed_model()
        if init_mode == "Natural language description":
            if not user_description.strip():
                st.error("Please describe your paper interests.")
                return
            with st.spinner("Embedding your interests..."):
                desc_embedding = model.embed_batch([user_description.strip()])[0]
            centroids = init_user_profile_from_description(desc_embedding)

        elif init_mode == "Category tags":
            if not selected_categories:
                st.error("Please select at least one topic.")
                return
            centroids = init_user_profile_from_topics(
                selected_categories,
                index.category_centroids,
            )

        else:
            if not scholar_url.strip():
                st.error("Please enter your Google Scholar URL.")
                return
            user_id = parse_scholar_url(scholar_url.strip())
            if not user_id:
                st.error("Invalid Google Scholar profile URL.")
                return

            with st.spinner("Fetching Scholar papers..."):
                try:
                    papers, _ = fetch_scholar_papers(
                        user_id,
                        max_papers=100,
                        sort_by_pubdate=False,
                    )
                except Exception:
                    papers = []

            if not papers:
                st.error("Could not load papers from this Scholar profile.")
                return

            top3 = sorted(papers, key=lambda p: p.get("citations", 0), reverse=True)[:3]
            if not top3:
                st.error("No papers found in the Scholar profile.")
                return

            with st.spinner("Fetching descriptions for top Scholar papers..."):
                papers_with_abstract = enrich_scholar_papers_with_descriptions(top3)
            with st.spinner("Embedding top Scholar papers..."):
                scholar_embeddings = model.embed_papers(papers_with_abstract)
            centroids = init_user_profile_from_scholar_top3(
                scholar_embeddings,
                index.category_centroids,
            )

        k_u = centroids.shape[0]
        user_id = create_user(name.strip(), centroids, k_u, diversity)

        st.session_state["user_id"] = user_id
        st.session_state["user_centroids"] = centroids
        st.session_state["user_k_u"] = k_u
        st.session_state["user_diversity"] = diversity
        st.session_state["onboarded"] = True
        st.rerun()
