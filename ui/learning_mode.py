# ui/learning_mode.py

import streamlit as st

from ai.workspace_cache import (
    read_workspace_summary_cache,
    write_workspace_connections_cache,
    write_workspace_summary_cache,
)
from ai.workspace_connections import generate_workspace_connections
from ai.workspace_summary import summarize_workspace
from ai.workspace_summary import resolve_summary_model
from pipeline.index import PaperIndex
from recommender.concept_map import (
    build_workspace_concept_map,
    make_workspace_concept_map_figure,
)
from recommender.engine import recommend
from user.db import get_saved_papers, get_seen_ids, log_feedback, update_centroids
from user.profile import apply_feedback, init_user_profile_v2, make_scholar_seed
from user.session import save_centroids_to_session
from ui.components import loading_spinner_with_message


WORKSPACE_SIMILAR_LIMIT = 5
WORKSPACE_CONCEPT_MAP_LAYOUT_VERSION = "pca-ai-connections-v1"


def _init_workspace_state():
    if "learning_workspace" not in st.session_state:
        st.session_state["learning_workspace"] = []

    if "right_sidebar_collapsed" not in st.session_state:
        st.session_state["right_sidebar_collapsed"] = False


def _enrich_saved_papers(saved_items, index):
    papers = []
    for item in saved_items:
        arxiv_id = item["arxiv_id"]
        paper = next((p for p in index.paper_meta if p["id"] == arxiv_id), None)

        papers.append(
            {
                **item,
                "title": paper.get("title", arxiv_id) if paper else arxiv_id,
                "abstract": paper.get("abstract", "") if paper else "",
                "categories": paper.get("categories", "") if paper else "",
            }
        )

    return papers


def _workspace_signature(workspace_papers: list[dict]) -> tuple[str, ...]:
    return tuple(paper["arxiv_id"] for paper in workspace_papers)


def _clear_workspace_outputs() -> None:
    for key in (
        "workspace_summary",
        "workspace_summary_signature",
        "workspace_similar_papers",
        "workspace_similar_signature",
        "workspace_similar_requested",
        "workspace_concept_map",
        "workspace_concept_map_signature",
        "workspace_concept_map_params",
        "workspace_connections",
        "workspace_connections_signature",
        "workspace_connections_source",
        "workspace_pending_action",
        "workspace_result_view",
    ):
        st.session_state.pop(key, None)


def _sync_workspace_output_cache(workspace_papers: list[dict]) -> None:
    st.session_state["workspace_cache_signature"] = _workspace_signature(
        workspace_papers
    )


def _add_to_workspace(arxiv_id: str, stay_on_tab: str | None = None):
    if arxiv_id not in st.session_state["learning_workspace"]:
        st.session_state["learning_workspace"].insert(0, arxiv_id)

    if stay_on_tab:
        st.session_state["requested_tab"] = stay_on_tab


def _add_all_to_workspace(saved_papers: list[dict], stay_on_tab: str | None = None):
    workspace_ids = st.session_state["learning_workspace"]
    existing_ids = set(workspace_ids)
    new_ids = [
        paper["arxiv_id"]
        for paper in saved_papers
        if paper["arxiv_id"] not in existing_ids
    ]

    if new_ids:
        st.session_state["learning_workspace"] = new_ids + workspace_ids

    if stay_on_tab:
        st.session_state["requested_tab"] = stay_on_tab


def _remove_from_workspace(arxiv_id: str):
    st.session_state["learning_workspace"] = [
        pid for pid in st.session_state["learning_workspace"] if pid != arxiv_id
    ]


def _remove_all_from_workspace():
    st.session_state["learning_workspace"] = []
    _clear_workspace_outputs()


def _save_paper_to_folders(arxiv_id: str, index: PaperIndex) -> bool:
    user_id = st.session_state["user_id"]
    saved_ids = {paper["arxiv_id"] for paper in get_saved_papers(user_id)}
    if arxiv_id in saved_ids:
        return False

    centroids = st.session_state["user_centroids"]
    suggestions = st.session_state.get("workspace_similar_papers", [])
    meta = next((paper for paper in suggestions if paper.get("id") == arxiv_id), None)
    cluster_id = meta.get("cluster_id", 0) if meta else 0
    score = meta.get("rec_score", 0.0) if meta else 0.0

    log_feedback(user_id, arxiv_id, "save", cluster_id, score)

    paper_idx = _paper_index_by_id(index, arxiv_id)
    if paper_idx is not None:
        paper_emb = index.embeddings[paper_idx]
        new_centroids = apply_feedback(centroids, paper_emb, "save")
        update_centroids(user_id, new_centroids)
        save_centroids_to_session(new_centroids)

    if "responded" not in st.session_state:
        st.session_state["responded"] = set()
    st.session_state["responded"].add(arxiv_id)
    return True


def _open_in_research_mode(
    arxiv_id: str,
    index: PaperIndex | None = None,
    save_to_folders: bool = False,
):
    if save_to_folders and index is not None:
        _save_paper_to_folders(arxiv_id, index)

    st.session_state["active_arxiv_id"] = arxiv_id
    st.session_state["requested_tab"] = "Research Lab"
    st.session_state["active_tab_value"] = "Research Lab"
    st.session_state.pop("overlay_page", None)
    st.rerun()


def _paper_index_by_id(index: PaperIndex, arxiv_id: str) -> int | None:
    for i, meta in enumerate(index.paper_meta):
        if meta.get("id") == arxiv_id:
            return i
    return None


def _workspace_onboarding_seeds(index: PaperIndex, workspace_papers: list[dict]):
    seeds = []
    for paper in workspace_papers:
        arxiv_id = paper["arxiv_id"]
        paper_idx = _paper_index_by_id(index, arxiv_id)
        if paper_idx is None:
            continue

        embedding = index.embeddings[paper_idx].astype("float32")
        norm = float((embedding * embedding).sum() ** 0.5)
        if norm <= 1e-12:
            continue

        title = paper.get("title") or arxiv_id
        seeds.append(make_scholar_seed(title, embedding / norm))

    return seeds


def _find_workspace_similar_papers(index: PaperIndex, workspace_papers: list[dict]):
    workspace_ids = [paper["arxiv_id"] for paper in workspace_papers]
    seeds = _workspace_onboarding_seeds(index, workspace_papers)
    if not seeds:
        return []

    workspace_profile = init_user_profile_v2(seeds)
    user_id = st.session_state["user_id"]
    shown_ids = set(st.session_state.get("shown_ids") or set())
    excluded_ids = get_seen_ids(user_id) | shown_ids | set(workspace_ids)
    recs = recommend(
        workspace_profile.centroids,
        excluded_ids,
        index,
        diversity=st.session_state.get("user_diversity", 0.5),
        n=WORKSPACE_SIMILAR_LIMIT,
    )
    shown_ids.update(paper["id"] for paper in recs if paper.get("id"))
    st.session_state["shown_ids"] = shown_ids
    return recs


def _handle_workspace_suggestion_save(arxiv_id: str, index: PaperIndex) -> None:
    _save_paper_to_folders(arxiv_id, index)
    st.rerun()


def _handle_workspace_suggestion_add(arxiv_id: str, index: PaperIndex) -> None:
    _save_paper_to_folders(arxiv_id, index)
    _add_to_workspace(arxiv_id)
    st.rerun()


def _render_workspace_suggestion(
    paper: dict,
    index: PaperIndex,
    saved_ids: set[str],
) -> None:
    arxiv_id = paper["id"]
    is_saved = arxiv_id in saved_ids
    is_in_workspace = arxiv_id in st.session_state.get("learning_workspace", [])

    with st.container(border=True):
        title = " ".join(paper.get("title", arxiv_id).split())
        st.markdown(f"**{title}**")

        categories = paper.get("categories", [])
        if categories:
            st.caption(" ".join(f"`{cat}`" for cat in categories))

        abstract = paper.get("abstract", "")
        if abstract:
            st.write(abstract[:320] + ("..." if len(abstract) > 320 else ""))

        score = paper.get("raw_similarity", paper.get("rec_score"))
        if score is not None:
            st.caption(f"Workspace recommendation score: {float(score):.3f}")

        cols = st.columns(3)
        with cols[0]:
            if st.button(
                "Added" if is_in_workspace else "Add",
                key=f"workspace_similar_add_{arxiv_id}",
                width="stretch",
                disabled=is_in_workspace,
            ):
                _handle_workspace_suggestion_add(arxiv_id, index)
        with cols[1]:
            if st.button(
                "Saved" if is_saved else "Save",
                key=f"workspace_similar_save_{arxiv_id}",
                width="stretch",
                disabled=is_saved,
            ):
                _handle_workspace_suggestion_save(arxiv_id, index)
        with cols[2]:
            st.link_button(
                "PDF",
                f"https://arxiv.org/pdf/{arxiv_id}",
                width="stretch",
            )


def _get_openai_api_key() -> str | None:
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except (FileNotFoundError, KeyError):
        return None


def _workspace_summary_ready(workspace_papers: list[dict]) -> bool:
    return bool(
        st.session_state.get("workspace_summary")
        and st.session_state.get("workspace_summary_signature")
        == _workspace_signature(workspace_papers)
    )


def _load_workspace_summary(workspace_papers: list[dict]) -> None:
    signature = _workspace_signature(workspace_papers)
    if (
        st.session_state.get("workspace_summary")
        and st.session_state.get("workspace_summary_signature") == signature
    ):
        return

    api_key = _get_openai_api_key()
    st.session_state.pop("workspace_summary", None)
    st.session_state.pop("workspace_summary_signature", None)
    try:
        with loading_spinner_with_message():
            summary = summarize_workspace(
                workspace_papers,
                api_key=api_key,
            )
            st.session_state["workspace_summary"] = summary
            st.session_state["workspace_summary_signature"] = signature
            try:
                write_workspace_summary_cache(
                    workspace_papers,
                    summary,
                    model=resolve_summary_model(),
                )
            except Exception as cache_exc:
                st.warning(f"Could not write workspace summary cache: {cache_exc}")
    except ValueError as exc:
        st.warning(
            f"{exc} Add OPENAI_API_KEY to your environment or .streamlit/secrets.toml."
        )
    except Exception as exc:
        st.error(f"Could not summarize the workspace: {exc}")


def _load_workspace_similar_papers(
    index: PaperIndex,
    workspace_papers: list[dict],
) -> None:
    signature = _workspace_signature(workspace_papers)
    st.session_state.pop("workspace_similar_papers", None)
    st.session_state.pop("workspace_similar_signature", None)
    with loading_spinner_with_message():
        st.session_state["workspace_similar_papers"] = (
            _find_workspace_similar_papers(index, workspace_papers)
        )
        st.session_state["workspace_similar_signature"] = signature
        st.session_state["workspace_similar_requested"] = True


def _load_workspace_concept_map(
    index: PaperIndex,
    workspace_papers: list[dict],
) -> None:
    workspace_ids = [paper["arxiv_id"] for paper in workspace_papers]
    signature = _workspace_signature(workspace_papers)
    params = (
        WORKSPACE_CONCEPT_MAP_LAYOUT_VERSION,
        st.session_state.get("workspace_connections_signature"),
    )
    if (
        st.session_state.get("workspace_concept_map")
        and st.session_state.get("workspace_concept_map_signature") == signature
        and st.session_state.get("workspace_concept_map_params") == params
    ):
        return

    st.session_state.pop("workspace_concept_map", None)
    st.session_state.pop("workspace_concept_map_signature", None)
    st.session_state.pop("workspace_concept_map_params", None)
    with loading_spinner_with_message():
        connections, connection_source = _load_workspace_connections(workspace_papers)
        st.session_state["workspace_concept_map"] = build_workspace_concept_map(
            index,
            workspace_ids,
            connections=connections,
        )
        st.session_state["workspace_concept_map"]["summary"][
            "connections_source"
        ] = connection_source
        st.session_state["workspace_concept_map_signature"] = signature
        st.session_state["workspace_concept_map_params"] = (
            WORKSPACE_CONCEPT_MAP_LAYOUT_VERSION,
            st.session_state.get("workspace_connections_signature"),
        )


def _load_workspace_connections(
    workspace_papers: list[dict],
) -> tuple[list[dict], str | None]:
    signature = _workspace_signature(workspace_papers)
    if (
        st.session_state.get("workspace_connections") is not None
        and st.session_state.get("workspace_connections_signature") == signature
    ):
        return (
            st.session_state.get("workspace_connections", []),
            st.session_state.get("workspace_connections_source"),
        )

    summary_text = read_workspace_summary_cache(workspace_papers)
    abstract_only = False
    source = "summary"
    if summary_text is None:
        if _workspace_summary_ready(workspace_papers):
            abstract_only = True
            source = "abstracts"
        else:
            return [], None

    try:
        connections = generate_workspace_connections(
            workspace_papers,
            summary_text,
            api_key=_get_openai_api_key(),
            abstract_only=abstract_only,
        )
    except ValueError as exc:
        st.warning(
            f"{exc} Add OPENAI_API_KEY to your environment or .streamlit/secrets.toml."
        )
        connections = []
    except Exception as exc:
        st.warning(f"Could not generate workspace connections: {exc}")
        connections = []

    st.session_state["workspace_connections"] = connections
    st.session_state["workspace_connections_signature"] = signature
    st.session_state["workspace_connections_source"] = source
    try:
        write_workspace_connections_cache(workspace_papers, connections)
    except Exception:
        pass
    return connections, source


def _render_map_summary(graph: dict) -> None:
    summary = graph.get("summary", {})
    counts = graph.get("counts", {})

    cols = st.columns(2)
    with cols[0]:
        st.metric("Papers", counts.get("papers", 0))
    with cols[1]:
        st.metric("Notable Connections", counts.get("connections", 0))

    if summary.get("connections_source") == "abstracts":
        st.caption("[ran on only abstracts]")


def _render_paper_node_key(graph: dict) -> None:
    paper_nodes = [node for node in graph.get("nodes", []) if node.get("type") == "paper"]
    if not paper_nodes:
        return
    key_text = "\n\n".join(
        f"**{node.get('label', '?')}** - {node.get('title', node.get('key', 'Paper'))}"
        for node in paper_nodes
    )
    st.markdown(key_text)


def _render_workspace_result_panel(
    index: PaperIndex,
    workspace_papers: list[dict],
    saved_ids: set[str],
) -> None:
    active_view = st.session_state.get("workspace_result_view")
    if not active_view:
        return

    st.divider()

    if active_view == "summary":
        summary = st.session_state.get("workspace_summary")
        if summary:
            st.subheader("Summary")
            st.markdown(summary)
        return

    if active_view == "similar":
        similar_papers = st.session_state.get("workspace_similar_papers", [])
        if similar_papers:
            st.subheader("Similar Papers")
            for paper in similar_papers:
                _render_workspace_suggestion(paper, index, saved_ids)
        elif st.session_state.get("workspace_similar_requested"):
            st.info("No new similar papers were found for this workspace.")
        return

    if active_view == "visualization":
        st.subheader("Visualize")
        if not _workspace_summary_ready(workspace_papers):
            st.info("Run Summarize before opening the visualization.")
            return
        _load_workspace_concept_map(
            index,
            workspace_papers,
        )
        graph = st.session_state.get("workspace_concept_map")
        if not graph or not graph.get("nodes"):
            st.info("Add papers to the workspace to build a concept map.")
            return

        _render_map_summary(graph)
        _render_paper_node_key(graph)
        try:
            fig = make_workspace_concept_map_figure(graph)
            st.plotly_chart(fig, width="stretch")
        except Exception as exc:
            st.warning(f"Could not render workspace concept map: {exc}")
            return

        counts = graph.get("counts", {})
        st.caption(
            "Nodes: "
            f"{counts.get('papers', 0)} papers, "
            f"{counts.get('connections', 0)} AI connections."
        )


def _render_workspace(index: PaperIndex, paper_lookup):
    workspace_ids = st.session_state["learning_workspace"]
    workspace_papers = [
        paper_lookup[pid] for pid in workspace_ids if pid in paper_lookup
    ]
    _sync_workspace_output_cache(workspace_papers)

    st.title("Workspace")

    heading_cols = st.columns([0.76, 0.24], vertical_alignment="center")
    with heading_cols[0]:
        st.subheader("Added Papers")
    with heading_cols[1]:
        if st.button(
            "Remove All",
            key="remove_all_workspace",
            width="stretch",
            disabled=not workspace_papers,
        ):
            _remove_all_from_workspace()
            st.rerun()

    if not workspace_papers:
        st.info("Add saved papers from the Papers folder to start a workspace.")
    else:
        with st.container(height=440, border=False):
            for paper in workspace_papers:
                with st.container(border=True):
                    title = paper.get("title", paper["arxiv_id"])
                    st.markdown(f"**{title}**")
                    st.caption(paper["arxiv_id"])

                    abstract = paper.get("abstract", "")
                    if abstract:
                        st.caption(
                            abstract[:260] + ("..." if len(abstract) > 260 else "")
                        )

                    cols = st.columns(3)
                    with cols[0]:
                        if st.button(
                            "Remove",
                            key=f"remove_workspace_{paper['arxiv_id']}",
                            width="stretch",
                        ):
                            _remove_from_workspace(paper["arxiv_id"])
                            st.rerun()

                    with cols[1]:
                        if st.button(
                            "Research",
                            key=f"workspace_research_{paper['arxiv_id']}",
                            width="stretch",
                        ):
                            _open_in_research_mode(paper["arxiv_id"])

                    with cols[2]:
                        st.link_button(
                            "PDF",
                            f"https://arxiv.org/pdf/{paper['arxiv_id']}",
                            width="stretch",
                        )

    st.divider()

    pending_action = st.session_state.get("workspace_pending_action")
    action_running = pending_action in {"see_more", "summary", "visualization"}
    summary_ready = _workspace_summary_ready(workspace_papers)

    with st.container(key="workspace_action_bar"):
        action_cols = st.columns(3)
        with action_cols[0]:
            if st.button(
                "See More",
                key="workspace_see_more",
                width="stretch",
                disabled=not workspace_papers or action_running,
                type="primary",
            ):
                st.session_state["workspace_pending_action"] = "see_more"
                st.session_state.pop("workspace_result_view", None)
                st.rerun()
        with action_cols[1]:
            if st.button(
                "Summarize",
                key="workspace_summarize",
                width="stretch",
                disabled=not workspace_papers or action_running,
                type="primary",
            ):
                st.session_state["workspace_pending_action"] = "summary"
                st.session_state.pop("workspace_result_view", None)
                st.rerun()
        with action_cols[2]:
            if st.button(
                "Visualization",
                key="workspace_visualize",
                width="stretch",
                disabled=not workspace_papers or action_running or not summary_ready,
                type="primary",
                help=(
                    None
                    if summary_ready
                    else "Run Summarize before opening the visualization."
                ),
            ):
                st.session_state["workspace_pending_action"] = "visualization"
                st.session_state.pop("workspace_result_view", None)
                st.rerun()

    if pending_action == "see_more":
        _load_workspace_similar_papers(index, workspace_papers)
        st.session_state.pop("workspace_pending_action", None)
        st.session_state["workspace_result_view"] = "similar"
        st.rerun()

    if pending_action == "summary":
        _load_workspace_summary(workspace_papers)
        st.session_state.pop("workspace_pending_action", None)
        if st.session_state.get("workspace_summary"):
            st.session_state["workspace_result_view"] = "summary"
        st.rerun()

    if pending_action == "visualization":
        _load_workspace_concept_map(index, workspace_papers)
        st.session_state.pop("workspace_pending_action", None)
        st.session_state["workspace_result_view"] = "visualization"
        st.rerun()

    _render_workspace_result_panel(index, workspace_papers, set(paper_lookup))


def _toggle_right_sidebar():
    st.session_state["right_sidebar_collapsed"] = not st.session_state[
        "right_sidebar_collapsed"
    ]
    st.rerun()


def _inject_right_sidebar_styles():
    is_collapsed = st.session_state["right_sidebar_collapsed"]
    rail_width = "3.75rem" if is_collapsed else "21rem"
    content_padding = "6.25rem" if is_collapsed else "24rem"

    st.markdown(
        """<style>
        .st-key-workspace_right_sidebar {
            position: fixed !important;
            top: 0;
            right: 0;
            width: %s;
            height: 100vh;
            padding: 5.25rem 1rem 1rem;
            background: rgb(240, 242, 246);
            border-left: 1px solid rgba(49, 51, 63, 0.18);
            overflow-y: auto;
            z-index: 999;
        }

        .st-key-workspace_right_sidebar_collapsed {
            padding-left: 0.55rem;
            padding-right: 0.55rem;
        }

        .st-key-workspace_right_sidebar button[kind='secondary'] {
            font-size: 1rem !important;
            font-weight: 600 !important;
            padding: 0.6rem 1rem !important;
            border-radius: 0.5rem !important;
            text-align: left !important;
            justify-content: flex-start !important;
        }

        .st-key-workspace_right_sidebar div[data-testid='stButton'] button p {
            text-align: left !important;
        }

        .st-key-collapse_right_sidebar button,
        .st-key-expand_right_sidebar button {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }

        .st-key-collapse_right_sidebar button div[data-testid='stMarkdownContainer'],
        .st-key-expand_right_sidebar button div[data-testid='stMarkdownContainer'] {
            display: flex !important;
            justify-content: center !important;
            width: 100%% !important;
        }

        .st-key-collapse_right_sidebar button p,
        .st-key-expand_right_sidebar button p {
            text-align: center !important;
            margin: 0 auto !important;
            width: 100%% !important;
        }

        .st-key-workspace_right_sidebar div[data-testid='stExpander'] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }

        .st-key-workspace_right_sidebar div[data-testid='stExpander'] details {
            border: none !important;
        }

        .st-key-workspace_right_sidebar div[data-testid='stExpander'] summary {
            font-size: 1rem !important;
            font-weight: 700 !important;
        }

        .stMainBlockContainer,
        [data-testid='stMainBlockContainer'],
        .block-container {
            padding-right: %s !important;
            padding-left: 2rem !important;
            max-width: none !important;
            width: 100%% !important;
        }

        @media (max-width: 1000px) {
            .st-key-workspace_right_sidebar {
                position: static !important;
                width: 100%%;
                height: auto;
                padding: 1rem 0 0;
                border-left: none;
                background: transparent;
            }

            .stMainBlockContainer,
            [data-testid='stMainBlockContainer'],
            .block-container {
                padding-right: 1rem !important;
                padding-left: 1rem !important;
            }
        }
        </style>"""
        % (rail_width, content_padding),
        unsafe_allow_html=True,
    )


def _render_saved_paper_row(paper, active_tab: str, row_index: int):
    arxiv_id = paper["arxiv_id"]
    title = paper.get("title", arxiv_id)
    user_id = st.session_state["user_id"]
    row_key = f"{row_index}_{arxiv_id}"
    is_active_research_paper = (
        active_tab == "Research Lab"
        and st.session_state.get("active_arxiv_id") == arxiv_id
        and st.session_state.get("overlay_page") is None
    )

    st.markdown(f"**{title[:80]}**")

    if active_tab == "Daily Feed":
        return

    # Layout for actions
    cols = st.columns([0.38, 0.34, 0.28])
    
    with cols[0]:
        is_added = arxiv_id in st.session_state.get("learning_workspace", [])
        if st.button(
            "Added" if is_added else "Add",
            key=f"sidebar_add_{row_key}",
            width="stretch",
            disabled=is_added,
        ):
            _add_to_workspace(arxiv_id, stay_on_tab=active_tab)
            st.rerun()

    with cols[1]:
        if st.button(
            "Read",
            key=f"sidebar_read_{row_key}",
            width="stretch",
            disabled=is_active_research_paper,
            help=(
                "This paper is open in Research Lab."
                if is_active_research_paper
                else "Open this paper in Research Lab."
            ),
        ):
            _open_in_research_mode(arxiv_id)

    with cols[2]:
        # The new Delete button
        if st.button("🗑️", key=f"sidebar_delete_{arxiv_id}", width="stretch", help="Remove from saved papers"):
            from user.db import delete_saved_paper
            delete_saved_paper(user_id, arxiv_id)
            
            # Clean up workspace if it was there
            if arxiv_id in st.session_state.get("learning_workspace", []):
                st.session_state["learning_workspace"].remove(arxiv_id)
                
            st.toast(f"Removed {arxiv_id}")
            st.rerun()


def _render_right_sidebar(saved_papers, active_tab: str):
    with st.container(key="workspace_right_sidebar"):
        collapsed = st.session_state["right_sidebar_collapsed"]

        if collapsed:
            if st.button("<", key="expand_right_sidebar", help="Expand folders"):
                _toggle_right_sidebar()
            return

        heading, collapse = st.columns([0.78, 0.22], vertical_alignment="center")
        with heading:
            st.subheader("Folders")
        with collapse:
            if st.button(">", key="collapse_right_sidebar", help="Collapse folders"):
                _toggle_right_sidebar()

        with st.expander(f"Papers ({len(saved_papers)})", expanded=True):
            if not saved_papers:
                st.caption("No saved papers yet.")
            else:
                workspace_ids = set(st.session_state.get("learning_workspace", []))
                all_added = all(
                    paper["arxiv_id"] in workspace_ids for paper in saved_papers
                )
                if active_tab != "Daily Feed":
                    if st.button(
                        "Add All",
                        key="sidebar_add_all_to_workspace",
                        width="stretch",
                        disabled=all_added,
                    ):
                        _add_all_to_workspace(saved_papers, stay_on_tab=active_tab)
                        st.rerun()
                    st.divider()

                for row_index, paper in enumerate(saved_papers):
                    _render_saved_paper_row(paper, active_tab, row_index)
                    st.divider()


def render_learning_mode(index):
    _init_workspace_state()

    user_id = st.session_state["user_id"]
    saved = get_saved_papers(user_id)
    saved_papers = _enrich_saved_papers(saved, index)
    paper_lookup = {paper["arxiv_id"]: paper for paper in saved_papers}

    _render_workspace(index, paper_lookup)

def render_workspace_sidebar(index, active_tab: str):
    _init_workspace_state()
    _inject_right_sidebar_styles()

    user_id = st.session_state["user_id"]
    saved = get_saved_papers(user_id)
    
    # FIX: Deduplicate papers by arxiv_id before processing
    seen_ids = set()
    unique_saved = []
    for item in saved:
        if item["arxiv_id"] not in seen_ids:
            unique_saved.append(item)
            seen_ids.add(item["arxiv_id"])
    
    # Use the unique list for enrichment
    saved_papers = _enrich_saved_papers(unique_saved, index)

    _render_right_sidebar(saved_papers, active_tab)
