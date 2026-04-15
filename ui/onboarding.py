"""Onboarding page: topic selection + optional Scholar profile + diversity slider."""

from __future__ import annotations

import streamlit as st
import numpy as np

from pipeline.index import PaperIndex
from pipeline.embed import EmbeddingModel
from pipeline.scholar_parser import load_scholar_papers
from user.db import create_user
from user.profile import init_user_profile
from ui.components import topic_selector


@st.cache_resource
def _get_embed_model() -> EmbeddingModel:
    return EmbeddingModel()


def render_onboarding(index: PaperIndex, db_path: str) -> None:
    st.title("ArXiv Daily")
    st.write("Personalized paper recommendations from arXiv, delivered daily.")
    st.divider()

    name = st.text_input("Your name", placeholder="Enter your display name")

    st.write("**Pick topics you're interested in:**")
    selected_categories = topic_selector(index.category_centroids)

    # -- Optional Scholar profile --
    st.write("**Optional:** paste your Google Scholar profile URL for "
             "more precise first-day recommendations.")
    scholar_url = st.text_input(
        "Google Scholar URL",
        placeholder="https://scholar.google.com/citations?user=...",
    )

    # -- Diversity slider --
    st.write("**How broad should your daily papers be?**")
    diversity = st.slider(
        "Diversity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="0 = focused on your strongest interest · 1 = explore broadly",
    )

    if st.button("Start reading", type="primary"):
        if not name.strip():
            st.error("Please enter your name.")
            return
        if not selected_categories:
            st.error("Please select at least one topic.")
            return

        # Embed Scholar papers if provided
        paper_embeddings = None
        if scholar_url.strip():
            with st.spinner("Fetching your Scholar profile..."):
                papers = load_scholar_papers(scholar_url.strip())
            if papers:
                with st.spinner("Embedding your papers..."):
                    model = _get_embed_model()
                    paper_embeddings = model.embed_papers(papers)
            else:
                st.warning("Could not load Scholar profile. "
                           "Continuing with topics only.")

        centroids = init_user_profile(
            selected_categories,
            index.category_centroids,
            paper_embeddings=paper_embeddings,
        )
        k_u = centroids.shape[0]
        user_id = create_user(name.strip(), centroids, k_u, diversity)

        st.session_state["user_id"] = user_id
        st.session_state["user_centroids"] = centroids
        st.session_state["user_k_u"] = k_u
        st.session_state["user_diversity"] = diversity
        st.session_state["onboarded"] = True
        st.rerun()
