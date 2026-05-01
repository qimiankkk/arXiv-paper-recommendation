"""Query-search UI for the main Daily Feed page."""

from __future__ import annotations

from functools import partial

import streamlit as st

from pipeline.embed import EmbeddingModel
from pipeline.index import PaperIndex
from recommender.query_search import expand_query, search_papers
from user.db import get_seen_ids, log_feedback, update_centroids
from user.profile import apply_feedback
from user.session import save_centroids_to_session
from ui.components import loading_spinner_with_message, paper_card


QUERY_SEARCH_EXAMPLES = [
    "retrieval-augmented generation for factual and citation-grounded LLMs",
    "diffusion models for medical image reconstruction with limited labels",
    "graph neural networks for molecular property prediction and drug discovery",
    "reinforcement learning for safe robotic navigation in uncertain environments",
    "neural operators for weather forecasting and climate downscaling",
    "multimodal learning across text, image, audio, and video",
    "privacy-preserving federated learning for distributed healthcare data",
    "interpretable machine learning for genomics and single-cell data",
    "AI alignment methods for scalable oversight and reward modeling",
    "time series forecasting for financial and macroeconomic data",
    "representation learning for neural population activity and behavior",
    "efficient transformers for sparse attention and long-context inference",
    "causal inference for treatment effects in observational healthcare data",
    "formal verification and program synthesis for code correctness",
    "autonomous driving perception and planning under uncertainty",
    "recommender systems balancing personalization, diversity, and fairness",
    "variational quantum algorithms and quantum machine learning",
    "speech recognition for noisy multilingual low-resource settings",
    "optimization methods for sharpness-aware and generalizable deep learning",
]


def _clear_query_search_state() -> None:
    for key in (
        "query_search_input",
        "query_search_time_filter",
        "query_search_options_open",
        "query_search_query",
        "query_search_expanded_query",
        "query_search_results",
        "query_search_clear_requested",
    ):
        st.session_state.pop(key, None)


def _rotating_search_placeholder() -> str:
    idx = st.session_state.get("query_search_example_idx", 0)
    st.session_state["query_search_example_idx"] = idx + 1
    return QUERY_SEARCH_EXAMPLES[idx % len(QUERY_SEARCH_EXAMPLES)]


@st.cache_resource(show_spinner=False)
def _get_query_embed_model() -> EmbeddingModel:
    return EmbeddingModel()


def _handle_search_feedback(arxiv_id: str, signal: str, index: PaperIndex) -> None:
    user_id = st.session_state["user_id"]
    centroids = st.session_state["user_centroids"]
    search_results = st.session_state.get("query_search_results", [])
    meta = next((r for r in search_results if r["id"] == arxiv_id), None)
    cluster_id = meta["cluster_id"] if meta else 0
    score = meta.get("search_score", 0.0) if meta else 0.0

    paper_idx = None
    for i, paper_meta in enumerate(index.paper_meta):
        if paper_meta["id"] == arxiv_id:
            paper_idx = i
            break

    log_feedback(user_id, arxiv_id, signal, cluster_id, score)

    if paper_idx is not None:
        paper_emb = index.embeddings[paper_idx]
        new_centroids = apply_feedback(centroids, paper_emb, signal)
        update_centroids(user_id, new_centroids)
        save_centroids_to_session(new_centroids)

    # --- NEW: update UI state ---
    if signal == "like":
        liked = set(st.session_state.get("liked", set()))
        liked.add(arxiv_id)
        st.session_state["liked"] = liked

    elif signal == "save":
        saved = set(st.session_state.get("saved", set()))
        saved.add(arxiv_id)
        st.session_state["saved"] = saved

    elif signal == "skip":
        skipped = set(st.session_state.get("skipped", set()))
        skipped.add(arxiv_id)
        st.session_state["skipped"] = skipped
    st.rerun()


def render_query_search(index: PaperIndex) -> bool:
    user_id = st.session_state["user_id"]

    st.markdown("**Search papers by topic, method, dataset, or research question...**")

    query = st.text_input(
        "Search papers",
        key="query_search_input",
        placeholder="Try: diffusion models for medical imaging",
        label_visibility="collapsed",
    )

    submitted = st.button("Search", key="query_search_submit")

    if submitted and query.strip():
        expanded = expand_query(query)

        with loading_spinner_with_message():
            model = _get_query_embed_model()
            q_emb = model.embed_query(expanded)

            results = search_papers(
                query=query,
                query_embedding=q_emb,
                user_centroids=st.session_state["user_centroids"],
                index=index,
                seen_ids=get_seen_ids(user_id),
                diversity=st.session_state["user_diversity"],
                n=20,
            )

        st.session_state["query_search_query"] = query
        st.session_state["query_search_results"] = results

    results = st.session_state.get("query_search_results", [])
    if not results:
        return False

    st.subheader(f"Results for: {st.session_state.get('query_search_query','')}")

    if st.button("Back to feed"):
        st.session_state.pop("query_search_results", None)
        st.rerun()

    # precompute state sets (avoids repeated lookup)
    liked = st.session_state.get("liked", set())
    saved = st.session_state.get("saved", set())
    skipped = st.session_state.get("skipped", set())

    for meta in results:
        arxiv_id = meta["id"]

        paper_card(
            meta,
            on_like=partial(_handle_search_feedback, signal="like", index=index),
            on_save=partial(_handle_search_feedback, signal="save", index=index),
            on_skip=partial(_handle_search_feedback, signal="skip", index=index),
            liked=arxiv_id in liked,
            saved=arxiv_id in saved,
            skipped=arxiv_id in skipped,
        )

    return True
