"""ArXiv Daily — Streamlit entry point.

Thin shell that initializes the database, loads the paper index (cached),
and routes to the appropriate page based on onboarding state.
"""

import base64

from pipeline.runtime import configure_single_thread_runtime

configure_single_thread_runtime()

from pathlib import Path

import streamlit as st

from pipeline.index import PaperIndex
from user.db import get_user, init_db
from user.session import load_or_init_session, is_onboarded, logout_user
from ui.onboarding import render_onboarding
from ui.daily_feed import render_daily_feed
from ui.research_mode import render_research_mode
from ui.profile_page import render_profile_page
from ui.learning_mode import render_learning_mode, render_workspace_sidebar

DB_PATH = "data/arxiv_rec.db"
MAIN_TABS = ("Daily Feed", "Workspace", "Research Lab")


def _clear_query_search_state() -> None:
    for key in (
        "query_search_input",
        "query_search_time_filter",
        "query_search_options_open",
        "query_search_query",
        "query_search_expanded_query",
        "query_search_results",
    ):
        st.session_state.pop(key, None)


def _normalize_tab(tab: str | None) -> str:
    return tab if tab in MAIN_TABS else "Daily Feed"


def _sync_active_tab_state() -> str:
    if "active_tab_value" not in st.session_state:
        st.session_state["active_tab_value"] = _normalize_tab(
            st.session_state.pop("active_tab", "Daily Feed")
        )

    if "requested_tab" in st.session_state:
        st.session_state["active_tab_value"] = _normalize_tab(
            st.session_state.pop("requested_tab")
        )

    st.session_state["active_tab_value"] = _normalize_tab(
        st.session_state["active_tab_value"]
    )
    return st.session_state["active_tab_value"]


def _activate_main_tab(tab: str) -> None:
    st.session_state["active_tab_value"] = tab
    st.session_state.pop("overlay_page", None)

    if tab == "Daily Feed":
        _clear_query_search_state()


st.set_page_config(
    page_title="Folio",
    page_icon="📄",
    layout="centered",
)

# -- global stylesheet -----------------------------------------------------
_css_path = Path(__file__).parent / "ui" / "style.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_index() -> PaperIndex:
    """Load the paper index once and cache it across Streamlit reruns.

    The ~6 GB embedding matrix must be loaded exactly once per process.
    st.cache_resource ensures this singleton behavior.

    Returns:
        A loaded PaperIndex instance.
    """
    idx = PaperIndex()
    idx.load()
    return idx


# -- init ------------------------------------------------------------------
init_db(DB_PATH)
load_or_init_session(DB_PATH)
index = load_index()

# -- check if pipeline has run ---------------------------------------------
if not index.is_loaded():
    st.error(
        "Data not found. Run: python scripts/run_offline_pipeline.py --limit 50000"
    )
    st.stop()

# -- route -----------------------------------------------------------------
if not is_onboarded():
    render_onboarding(index, DB_PATH)
else:
    active_tab = _sync_active_tab_state()
    overlay = st.session_state.get("overlay_page")

    # 1. --- LEFT SIDEBAR (Native Navigation) ---
    with st.sidebar:
        user_id = st.session_state["user_id"]
        user_data = get_user(user_id)

        avatar_file = user_data.get("profile_pic") or "default_avatar.jpg"
        avatar_path = Path(".streamlit/static/avatars") / avatar_file

        if avatar_path.exists():
            with open(avatar_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            profile_html = (
            f'<div style="display:flex; flex-direction:column; align-items:center; justify-content:center; margin-top:10px; width:100%;">'
            f'<img src="data:image/png;base64,{img_b64}" '
            f'style="width:120px;height:120px;border-radius:50%;object-fit:cover;display:block;margin:0 auto;">'
            f'<div style="text-align:center;font-weight:600;margin-top:15px;margin-bottom:15px;font-size:4rem;">'
            f'{user_data["display_name"]}'
            f'</div>'
            f'</div>'
            )

            st.markdown(profile_html, unsafe_allow_html=True)

        else:
            st.markdown(
                f"""
        <div style="
            display:flex;
            flex-direction:column;
            align-items:center;
            margin-top:10px;
            width:100%;
        ">
            <div style="font-size:4rem;">&#129489;</div>
            <div style="font-weight:600; margin-top:8px; font-size:3rem;">
                {user_data['display_name']}
            </div>
        </div>
        """,
                unsafe_allow_html=True
            )

        if st.button(
            "User Profile",
            width="stretch",
            type="primary" if overlay == "profile" else "secondary",
        ):
            st.session_state["overlay_page"] = "profile"
            st.rerun()

        if st.button("Log out", width="stretch"):
            logout_user()
            st.session_state.clear()
            st.rerun()

        st.divider()

        for tab in MAIN_TABS:
            if st.button(
                tab,
                key=f"sidebar_nav_{tab.lower().replace(' ', '_')}",
                width="stretch",
                type="primary" if overlay is None and active_tab == tab else "secondary",
            ):
                _activate_main_tab(tab)
                st.rerun()
    # 2. --- MAIN CONTENT AREA ---
    # Right Sidebar (Folders/Files)
    render_workspace_sidebar(index, active_tab)

    # 3. --- ROUTING ---
    if overlay == "profile":
        render_profile_page(index)

    else:
        # Standard Mode Routing
        if active_tab == "Research Lab":
            render_research_mode(index)
        elif active_tab == "Workspace":
            render_learning_mode(index)
        else:
            render_daily_feed(index, DB_PATH)
