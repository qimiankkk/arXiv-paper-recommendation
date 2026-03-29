"""Onboarding page for new users.

Shown when the user has not yet completed the initial setup.
Collects a display name and topic preferences, then creates a user profile
with an initial embedding derived from the selected category centroids.
"""

from __future__ import annotations

import streamlit as st

from pipeline.index import PaperIndex
from user.db import create_user
from user.profile import init_embedding_from_topics
from ui.components import topic_selector


def render_onboarding(index: PaperIndex, db_path: str) -> None:
    """Render the onboarding page for a new (not yet onboarded) user.

    Args:
        index: The loaded PaperIndex (needed for category_centroids).
        db_path: Path to the SQLite database.
    """
    st.title("ArXiv Daily")
    st.write("Personalized paper recommendations from arXiv, delivered daily.")

    st.divider()

    name = st.text_input("Your name", placeholder="Enter your display name")

    st.write("**Pick topics you're interested in:**")
    selected_categories = topic_selector(index.category_centroids)

    if st.button("Start reading", type="primary"):
        if not name.strip():
            st.error("Please enter your name.")
            return
        if not selected_categories:
            st.error("Please select at least one topic.")
            return

        embedding = init_embedding_from_topics(
            selected_categories, index.category_centroids
        )
        user_id = create_user(name.strip(), embedding)

        st.session_state["user_id"] = user_id
        st.session_state["user_embedding"] = embedding
        st.session_state["onboarded"] = True
        st.rerun()
