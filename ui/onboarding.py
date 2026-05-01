"""Onboarding page: account setup, profile picture, and research interests."""

from __future__ import annotations
from datetime import date, datetime
from pathlib import Path
import numpy as np
import streamlit as st

from pipeline.concept_tags import CONCEPT_TAG_MAP
from pipeline.embed import EmbeddingModel
from pipeline.index import PaperIndex
from pipeline.interest_expander import embed_free_text_interests
from pipeline.scholar_parser import load_scholar_papers
from ui.components import (
    MAX_ONBOARDING_TAGS,
    TOPIC_LABELS,
    expand_topic_labels,
    free_text_input,
    loading_spinner_with_message,
    unified_tag_selector,
)
from user.db import create_user
from user.session import login_with_credentials
from user.profile import (
    SeedSignal,
    init_user_profile_v2,
    make_category_seed,
    make_concept_seed,
    make_freetext_seed,
    make_scholar_seed,
)

# Constants for avatar selection
AVATARS = [f"avatar_{i}.jpg" for i in range(1, 7)]

@st.cache_resource(show_spinner=False)
def _get_embed_model() -> EmbeddingModel:
    return EmbeddingModel()

def make_category_seeds_from_topic_labels(
    selected_labels: list[str],
    category_centroids: dict[str, np.ndarray],
) -> list[SeedSignal]:
    """Build category seeds from human-readable onboarding labels."""
    topic_keys = expand_topic_labels(selected_labels, TOPIC_LABELS, category_centroids)
    if selected_labels and not topic_keys:
        raise ValueError("Selected topics are not available in current centroids.")

    seeds: list[SeedSignal] = []
    for code in topic_keys:
        label = next((l for l in selected_labels if code in TOPIC_LABELS.get(l, [])), code)
        seeds.append(make_category_seed(code, label, category_centroids[code]))
    return seeds

def render_onboarding(index: PaperIndex, db_path: str) -> None:
    """Render the multi-page onboarding flow."""
    if "onboarding_step" not in st.session_state:
        st.session_state["onboarding_step"] = 1

    if st.session_state["onboarding_step"] == 1:
        render_step_one()
    else:
        render_step_two(index, db_path)

def render_step_one() -> None:
    """Step 1: Account setup and binary-safe avatar selection."""
    st.title("Folio")
    st.write("Your personal & personalized research tool.")
    st.divider()

    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = None

    mode = st.session_state["auth_mode"]
    hour = datetime.now().hour

    if 5 <= hour < 12:
        greeting = "Good morning! Let's get you started."
    elif 12 <= hour < 18:
        greeting = "Good afternoon! Let's get you started."
    elif 18 <= hour < 22:
        greeting = "Good evening! Let's get you started."
    else:
        greeting = "Hey there, night owl. Let's get you started."

    # Welcome / Mode Selection [cite: 3352, 4901]
    if mode is None:
        st.subheader(greeting)
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Log in", use_container_width=True):
                st.session_state["auth_mode"] = "Log in"
                st.rerun()
        with col2:
            if st.button("Sign up", use_container_width=True):
                st.session_state["auth_mode"] = "Create account"
                st.rerun()
        with col3:
            if st.button("Continue as guest", use_container_width=True):
                st.session_state["auth_mode"] = "Continue as guest"
                st.rerun()
        return

    # Handle Login [cite: 3362, 5048]
    if mode == "Log in":
        if st.button("← Back"):
            st.session_state["auth_mode"] = None
            st.rerun()
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Log in", type="primary"):
            if login_with_credentials(username, password):
                st.rerun()
            else:
                st.error("Invalid credentials.")
        return

    # Sign up / Guest setup [cite: 3368, 5050]
    if st.button("← Back"):
        st.session_state["auth_mode"] = None
        st.rerun()

    is_guest = mode == "Continue as guest"
    if not is_guest:
        username = st.text_input("Username", placeholder="Choose a username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm password", type="password")
        name = username.strip()
    else:
        st.caption("Guest mode: no account required. Your session is temporary.")
        username, password, confirm_password = "", "", ""
        name = st.text_input("Your name", placeholder="Enter your display name")

    if "selected_avatar" not in st.session_state:
        st.session_state["selected_avatar"] = AVATARS[0]

    cols = st.columns(6)
    avatar_dir = Path(".streamlit/static/avatars")
    
    for i, avatar_file in enumerate(AVATARS):
        with cols[i]:
            # BINARY READ FIX: This ensures local images show up 
            img_path = avatar_dir / avatar_file
            if img_path.exists():
                with open(img_path, "rb") as f:
                    st.image(f.read(), use_container_width=True)
            else:
                st.error("Missing image")
            
            is_selected = st.session_state.get("selected_avatar") == avatar_file
    
            if st.button(
                "Selected" if is_selected else "Select", 
                key=f"sel_{avatar_file}",
                type="primary" if is_selected else "secondary",
                use_container_width=True
            ):
                st.session_state["selected_avatar"] = avatar_file
                st.rerun()

    if st.button("Next: Research Interests", type="primary", use_container_width=True):
        if not is_guest and (not username or len(password) < 8 or password != confirm_password):
            st.error("Please verify your account details.")
        elif is_guest and not name:
            st.error("Please enter a name.")
        else:
            st.session_state["temp_creds"] = {
                "username": username, "password": password, "display_name": name,
                "profile_pic": st.session_state["selected_avatar"], "is_guest": is_guest
            }
            st.session_state["onboarding_step"] = 2
            st.rerun()

def render_step_two(index: PaperIndex, db_path: str) -> None:
    """Step 2: Research Interest selection."""
    st.title("What are you researching?")
    if st.button("← Back to Account Details"):
        st.session_state["onboarding_step"] = 1
        st.rerun()
    
    # Tag and Interest selection [cite: 3369, 5052]
    topic_labels, concept_keys = unified_tag_selector(index.category_centroids, index.concept_embeddings or {})
    free_texts = free_text_input()
    scholar_url = st.text_input("Google Scholar URL (optional)", placeholder="https://scholar.google.com/...")
    diversity = st.slider("Exploration range", 0.0, 1.0, 0.5, 0.1)

    if st.button("Start reading", type="primary", use_container_width=True):
        creds = st.session_state["temp_creds"]
        seeds: list[SeedSignal] = []

        # Build seeds from selections [cite: 3376, 5036]
        try:
            seeds.extend(make_category_seeds_from_topic_labels(topic_labels, index.category_centroids))
            for key in concept_keys:
                tag = CONCEPT_TAG_MAP[key]
                seeds.append(make_concept_seed(key, tag.label, index.concept_embeddings[key]))
            
            if free_texts:
                with loading_spinner_with_message():
                    model = _get_embed_model()
                    for phrase, emb in embed_free_text_interests(free_texts, model):
                        seeds.append(make_freetext_seed(phrase, emb))
            
            if not seeds:
                st.error("Please select at least one research interest.")
                return

            # Initialize profile [cite: 3383, 5061]
            result = init_user_profile_v2(seeds)
            user_id = create_user(
                display_name=creds["display_name"],
                centroids=result.centroids,
                k_u=result.centroids.shape[0],
                diversity=diversity,
                thread_weights=result.thread_weights,
                thread_labels=result.thread_labels,
                username=creds["username"] if not creds["is_guest"] else None,
                password=creds["password"] if not creds["is_guest"] else None,
                profile_pic=creds["profile_pic"] 
            )
            
            # Finalize session [cite: 3385, 5117]
            st.session_state.update({
                "user_id": user_id,
                "user_centroids": result.centroids,
                "user_k_u": result.centroids.shape[0],
                "user_diversity": diversity,
                "thread_labels": result.thread_labels,
                "thread_weights": result.thread_weights,
                "onboarded": True,
            })
            st.rerun()

        except Exception as e:
            st.error(f"Setup error: {e}")
