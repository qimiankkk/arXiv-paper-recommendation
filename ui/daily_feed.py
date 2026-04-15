"""Daily feed page for onboarded users.

Main page showing 5 recommended papers per day with like/save/skip actions.
Handles feedback processing: logging to DB, updating user centroids via EMA,
and persisting the updated centroids.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from pipeline.index import PaperIndex
from recommender.engine import recommend
from user.db import get_user, get_seen_ids, log_feedback, update_centroids
from user.profile import apply_feedback
from user.session import save_centroids_to_session
from ui.components import paper_card


def _handle_feedback(
    arxiv_id: str, signal: str, index: PaperIndex
) -> None:
    """Process a feedback event: log, update centroids, mark responded."""
    user_id = st.session_state["user_id"]
    centroids = st.session_state["user_centroids"]

    recs = st.session_state.get("todays_recs", [])
    meta = next((r for r in recs if r["id"] == arxiv_id), None)
    cluster_id = meta["cluster_id"] if meta else 0
    score = meta.get("rec_score", 0.0) if meta else 0.0

    paper_idx = None
    for i, pm in enumerate(index.paper_meta):
        if pm["id"] == arxiv_id:
            paper_idx = i
            break

    log_feedback(user_id, arxiv_id, signal, cluster_id, score)

    if paper_idx is not None:
        paper_emb = index.embeddings[paper_idx]
        new_centroids = apply_feedback(centroids, paper_emb, signal)
        update_centroids(user_id, new_centroids)
        save_centroids_to_session(new_centroids)

    if "responded" not in st.session_state:
        st.session_state["responded"] = set()
    st.session_state["responded"].add(arxiv_id)
    st.rerun()


def render_daily_feed(index: PaperIndex, db_path: str) -> None:
    """Render the daily paper feed for an onboarded user.

    Args:
        index: The loaded PaperIndex for recommendation lookups.
        db_path: Path to the SQLite database for feedback logging.
    """
    user_id = st.session_state["user_id"]
    user = get_user(user_id)

    st.title(f"Good morning, {user['display_name']}")
    st.caption(date.today().strftime("%A, %B %d, %Y"))

    with st.sidebar:
        st.write(f"**{user['display_name']}**")
        created = user["created_at"][:10] if user["created_at"] else ""
        st.caption(f"Member since {created}")
        seen = get_seen_ids(user_id)
        st.metric("Papers seen", len(seen))
        st.metric("Research threads", user["k_u"])
        st.metric("Diversity", f"{user['diversity']:.1f}")

    if "todays_recs" not in st.session_state:
        centroids = st.session_state["user_centroids"]
        diversity = st.session_state["user_diversity"]
        seen_ids = get_seen_ids(user_id)
        with st.spinner("Finding your papers..."):
            recs = recommend(centroids, seen_ids, index, diversity=diversity, n=5)
        st.session_state["todays_recs"] = recs

    if "responded" not in st.session_state:
        st.session_state["responded"] = set()

    recs = st.session_state["todays_recs"]

    if not recs:
        st.info("You've seen most papers in your areas — come back tomorrow.")
        return
    if len(recs) < 5:
        st.warning("You've seen most papers in your areas. Here's what we found:")

    st.subheader(f"Your {len(recs)} paper{'s' if len(recs) != 1 else ''} for today")

    for meta in recs:
        paper_card(
            meta,
            on_like=lambda aid: _handle_feedback(aid, "like", index),
            on_save=lambda aid: _handle_feedback(aid, "save", index),
            on_skip=lambda aid: _handle_feedback(aid, "skip", index),
        )

    responded = st.session_state.get("responded", set())
    rec_ids = {r["id"] for r in recs}
    if rec_ids and rec_ids.issubset(responded):
        st.success("Come back tomorrow for new recommendations!")
