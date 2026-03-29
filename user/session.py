"""Streamlit session state helpers bridging st.session_state and the database.

Manages the lifecycle of user state within a Streamlit session:
- Initializing default session values for new visitors.
- Loading an existing user from the DB into the session.
- Syncing embedding updates between session state and DB.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from user.db import get_user


def load_or_init_session(db_path: str) -> None:
    """Initialize Streamlit session state with default user values.

    Called once at app startup. If user state is not already present in
    st.session_state, sets the following keys:
        - "user_id": None
        - "user_embedding": None
        - "onboarded": False

    Args:
        db_path: Path to the SQLite database (stored for later use).
    """
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None
        st.session_state["user_embedding"] = None
        st.session_state["onboarded"] = False


def login_user(user_id: str, db_path: str) -> bool:
    """Load an existing user from the database into session state.

    Args:
        user_id: The UUID string of the user to load.
        db_path: Path to the SQLite database.

    Returns:
        True if the user was found and loaded, False otherwise.
    """
    user = get_user(user_id)
    if user is None:
        return False

    st.session_state["user_id"] = user["user_id"]
    st.session_state["user_embedding"] = user["embedding"]
    st.session_state["onboarded"] = True
    return True


def save_embedding_to_session(embedding: np.ndarray) -> None:
    """Update the user embedding in the current Streamlit session state.

    Args:
        embedding: The new embedding, shape (768,), float32, unit-norm.
    """
    st.session_state["user_embedding"] = embedding


def is_onboarded() -> bool:
    """Check whether the current session user has completed onboarding.

    Returns:
        True if st.session_state["onboarded"] is True, False otherwise.
    """
    return st.session_state.get("onboarded", False)
