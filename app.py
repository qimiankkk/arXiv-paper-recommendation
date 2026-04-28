"""ArXiv Daily — Streamlit entry point.

Thin shell that initializes the database, loads the paper index (cached),
and routes to the appropriate page based on onboarding state.
"""

from pipeline.runtime import configure_single_thread_runtime

configure_single_thread_runtime()

import streamlit as st

from pipeline.index import PaperIndex
from user.db import init_db
from user.session import load_or_init_session, is_onboarded
from ui.onboarding import render_onboarding
from ui.daily_feed import render_daily_feed

DB_PATH = "data/arxiv_rec.db"

st.set_page_config(
    page_title="ArXiv Daily",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource
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
    render_daily_feed(index, DB_PATH)
