"""Workspace PCA visualization with cluster centroids and saved papers."""

from __future__ import annotations

from collections import Counter, defaultdict
import textwrap
from pathlib import Path

import numpy as np

from pipeline.index import PaperIndex
from recommender.visualization import get_primary_category


VIZ_ARTIFACTS = (
    {
        "name": "PCA",
        "coords": Path("data/diagnostics/pca_cluster_viz_coords.npy"),
        "indices": Path("data/diagnostics/pca_cluster_viz_indices.npy"),
    },
)
CONNECTION_COLORS = {
    "thesis": "#7c3aed",
    "assumption": "#f59e0b",
    "methodology": "#059669",
    "dataset": "#2563eb",
    "evaluation": "#dc2626",
    "multiple": "#111827",
}
EDGE_HOVER_WRAP_WIDTH = 56
EDGE_HOVER_MARKER_STEPS = 13
PAPER_LABEL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _paper_index_by_id(index: PaperIndex, arxiv_id: str) -> int | None:
    for i, meta in enumerate(index.paper_meta):
        if meta.get("id") == arxiv_id:
            return i
    return None


def _paper_letter(index: int) -> str:
    if index < len(PAPER_LABEL_ALPHABET):
        return PAPER_LABEL_ALPHABET[index]
    quotient, remainder = divmod(index, len(PAPER_LABEL_ALPHABET))
    return f"{PAPER_LABEL_ALPHABET[remainder]}{quotient + 1}"


def _wrap_hover_text(text: str, width: int = EDGE_HOVER_WRAP_WIDTH) -> str:
    raw_text = str(text or "").strip()
    if not raw_text:
        return ""
    paragraphs = [
        " ".join(paragraph.split())
        for paragraph in raw_text.split("\n\n")
        if paragraph.strip()
    ]
    wrapped = [
        "<br>".join(
            textwrap.wrap(paragraph, width=width, break_long_words=False)
        )
        for paragraph in paragraphs
    ]
    return "<br>────────────────────<br>".join(wrapped)


def _combine_connection_edges(edges: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for edge in edges:
        pair = tuple(sorted([str(edge["source"]), str(edge["target"])]))
        grouped[pair].append(edge)

    combined: list[dict] = []
    for pair_edges in grouped.values():
        if len(pair_edges) == 1:
            combined.append(pair_edges[0])
            continue

        types = []
        names = []
        descriptions = []
        sections = []
        confidence = 0.0
        for edge in pair_edges:
            edge_type = str(edge.get("type") or "")
            if edge_type and edge_type not in types:
                types.append(edge_type)
            name = str(edge.get("name") or "")
            if name and name not in names:
                names.append(name)
            description = str(edge.get("description") or edge.get("rationale") or "")
            if description:
                descriptions.append(f"{edge_type.title()}: {description}")
            section = str(edge.get("summary_section") or "")
            if section and section not in sections:
                sections.append(section)
            confidence = max(confidence, float(edge.get("confidence", 0.0) or 0.0))

        first = pair_edges[0].copy()
        first.update(
            {
                "type": "multiple",
                "name": "Multiple connections",
                "connection_types": ", ".join(t.title() for t in types),
                "summary_section": ", ".join(sections),
                "rationale": "; ".join(names),
                "description": "\n\n".join(descriptions),
                "confidence": confidence,
            }
        )
        combined.append(first)
    return combined


def _cluster_centroid_names(index: PaperIndex) -> dict[int, str]:
    cluster_ids = getattr(index, "cluster_ids", None)
    paper_meta = getattr(index, "paper_meta", None)
    if cluster_ids is None or paper_meta is None:
        return {}

    category_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for paper_i, cluster_id in enumerate(np.asarray(cluster_ids).ravel()):
        if paper_i >= len(paper_meta):
            break
        category = get_primary_category(paper_meta[paper_i].get("categories", ""))
        category_counts[int(cluster_id)][category] += 1

    names: dict[int, str] = {}
    for cluster_id, counts in category_counts.items():
        category, count = counts.most_common(1)[0]
        names[cluster_id] = f"{category} neighborhood ({count} papers)"
    return names


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if len(coords) == 0:
        return coords
    coords = coords - coords.mean(axis=0, keepdims=True)
    scale = float(np.abs(coords).max())
    if scale > 1e-12:
        coords = coords / scale
    return coords


def _fallback_coords(vectors: np.ndarray) -> tuple[np.ndarray, str]:
    if len(vectors) == 0:
        return np.empty((0, 2), dtype=np.float32), "fallback"
    if len(vectors) == 1:
        return np.array([[0.0, 0.0]], dtype=np.float32), "fallback"
    if len(vectors) == 2:
        return np.array([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32), "fallback"

    from sklearn.decomposition import PCA

    coords = PCA(n_components=2, random_state=42).fit_transform(vectors)
    return coords.astype(np.float32), "fallback PCA"


def _estimate_pca_coords_from_sample(
    sample_vectors: np.ndarray,
    sample_coords: np.ndarray,
    target_vectors: np.ndarray,
    neighbors: int = 25,
) -> np.ndarray:
    if len(target_vectors) == 0:
        return np.empty((0, 2), dtype=np.float32)
    k = min(neighbors, len(sample_vectors))
    estimated: list[np.ndarray] = []
    for vector in target_vectors:
        sims = sample_vectors @ vector
        top = np.argpartition(sims, -k)[-k:]
        top_sims = sims[top]
        weights = np.exp((top_sims - top_sims.max()) * 20.0)
        weights = weights / np.maximum(weights.sum(), 1e-12)
        estimated.append((sample_coords[top] * weights[:, None]).sum(axis=0))
    return np.asarray(estimated, dtype=np.float32)


def _workspace_pca_coords(
    index: PaperIndex,
    paper_indices: list[int],
    paper_vectors: np.ndarray,
    centroid_vectors: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    centroid_vectors = (
        np.empty((0, paper_vectors.shape[1]), dtype=np.float32)
        if centroid_vectors is None
        else _unit_rows(centroid_vectors)
    )
    for artifact in VIZ_ARTIFACTS:
        if not artifact["coords"].exists() or not artifact["indices"].exists():
            continue

        coords = np.load(artifact["coords"])
        sampled_indices = np.load(artifact["indices"]).astype(np.int64)
        if len(coords) != len(sampled_indices):
            continue
        if len(sampled_indices) == 0:
            continue
        if int(sampled_indices.max()) >= len(index.embeddings):
            continue

        sample_vectors = _unit_rows(
            np.asarray(index.embeddings[sampled_indices], dtype=np.float32)
        )
        sampled_pos = {
            int(paper_idx): pos for pos, paper_idx in enumerate(sampled_indices)
        }
        paper_result: list[np.ndarray | None] = []
        missing: list[tuple[int, np.ndarray]] = []
        for pos, paper_idx in enumerate(paper_indices):
            coord_pos = sampled_pos.get(int(paper_idx))
            if coord_pos is None:
                paper_result.append(None)
                missing.append((pos, paper_vectors[pos]))
            else:
                paper_result.append(np.asarray(coords[coord_pos], dtype=np.float32))

        if missing:
            estimated = _estimate_pca_coords_from_sample(
                sample_vectors,
                coords,
                np.asarray([vector for _, vector in missing], dtype=np.float32),
            )
            for (pos, _), coord in zip(missing, estimated):
                paper_result[pos] = coord

        if all(coord is not None for coord in paper_result):
            paper_coords = np.stack(paper_result).astype(np.float32)
            centroid_coords = _estimate_pca_coords_from_sample(
                sample_vectors,
                coords,
                centroid_vectors,
            )
            all_coords = _normalize_coords(np.vstack([centroid_coords, paper_coords]))
            return (
                all_coords[len(centroid_coords) :],
                all_coords[: len(centroid_coords)],
                str(artifact["name"]),
            )

    all_vectors = np.vstack([centroid_vectors, paper_vectors])
    all_coords, source = _fallback_coords(all_vectors)
    all_coords = _normalize_coords(all_coords)
    return (
        all_coords[len(centroid_vectors) :],
        all_coords[: len(centroid_vectors)],
        source,
    )


def build_workspace_concept_map(
    index: PaperIndex,
    workspace_ids: list[str],
    connections: list[dict] | None = None,
    *_args,
    **_kwargs,
) -> dict:
    """Build a PCA-positioned map for the current Workspace."""
    paper_refs = [
        (arxiv_id, idx)
        for arxiv_id in workspace_ids
        if (idx := _paper_index_by_id(index, arxiv_id)) is not None
    ]
    paper_indices = [idx for _, idx in paper_refs]
    raw_centroids = getattr(index, "centroids", None)
    centroid_vectors = (
        np.asarray(raw_centroids, dtype=np.float32)
        if raw_centroids is not None and len(raw_centroids)
        else np.empty((0, np.asarray(index.embeddings).shape[1]), dtype=np.float32)
    )
    if not paper_indices:
        return {
            "nodes": [],
            "edges": [],
            "summary": {"position_source": None},
            "counts": {
                "papers": 0,
                "clusters": 0,
                "connections": 0,
                "concepts": 0,
            },
        }

    paper_vectors = _unit_rows(
        np.asarray(index.embeddings[paper_indices], dtype=np.float32)
    )
    paper_coords, centroid_coords, position_source = _workspace_pca_coords(
        index,
        paper_indices,
        paper_vectors,
        centroid_vectors,
    )
    paper_metas = [index.paper_meta[paper_idx] for paper_idx in paper_indices]
    centroid_names = _cluster_centroid_names(index)

    nodes: list[dict] = []
    for cluster_id, coord in enumerate(centroid_coords):
        name = centroid_names.get(cluster_id, f"Cluster centroid {cluster_id}")
        nodes.append(
            {
                "id": f"cluster:{cluster_id}",
                "key": str(cluster_id),
                "label": f"C{cluster_id}",
                "title": name,
                "type": "cluster_centroid",
                "cluster": int(cluster_id),
                "x": float(coord[0]),
                "y": float(coord[1]),
            }
        )

    for pos, meta in enumerate(paper_metas):
        arxiv_id = str(meta.get("id", paper_refs[pos][0]))
        title = meta.get("title", arxiv_id)
        cluster_ids = getattr(index, "cluster_ids", None)
        cluster_id = (
            int(cluster_ids[paper_indices[pos]]) if cluster_ids is not None else -1
        )
        nodes.append(
            {
                "id": f"paper:{arxiv_id}",
                "key": arxiv_id,
                "label": _paper_letter(pos),
                "title": title,
                "type": "paper",
                "primary_category": get_primary_category(meta.get("categories", "")),
                "cluster": cluster_id,
                "x": float(paper_coords[pos, 0]),
                "y": float(paper_coords[pos, 1]),
            }
        )

    paper_nodes_by_key = {
        node["key"]: node for node in nodes if node.get("type") == "paper"
    }
    edges: list[dict] = []
    for edge_i, connection in enumerate(connections or []):
        source = str(connection.get("source") or "")
        target = str(connection.get("target") or "")
        connection_type = str(connection.get("type") or "")
        if (
            source not in paper_nodes_by_key
            or target not in paper_nodes_by_key
            or connection_type not in CONNECTION_COLORS
        ):
            continue
        edges.append(
            {
                "id": f"connection:{edge_i}",
                "source": f"paper:{source}",
                "target": f"paper:{target}",
                "source_title": paper_nodes_by_key[source].get("title", source),
                "target_title": paper_nodes_by_key[target].get("title", target),
                "type": connection_type,
                "name": str(connection.get("name") or connection_type.title()),
                "summary_section": str(connection.get("summary_section") or ""),
                "rationale": str(connection.get("rationale") or ""),
                "description": str(
                    connection.get("description")
                    or connection.get("overview")
                    or connection.get("rationale")
                    or ""
                ),
                "confidence": float(connection.get("confidence", 0.0) or 0.0),
            }
        )

    edges = _combine_connection_edges(edges)

    return {
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "position_source": position_source,
        },
        "counts": {
            "papers": len(paper_metas),
            "clusters": len(centroid_coords),
            "connections": len(edges),
            "concepts": 0,
        },
    }


def make_workspace_concept_map_figure(graph: dict):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is required for Workspace Visualization. Install dependencies "
            "with `pip install -r requirements.txt`."
        ) from exc

    centroid_nodes = [
        node for node in graph.get("nodes", []) if node.get("type") == "cluster_centroid"
    ]
    paper_nodes = [node for node in graph.get("nodes", []) if node.get("type") == "paper"]
    paper_nodes_by_id = {node["id"]: node for node in paper_nodes}
    fig = go.Figure()

    if centroid_nodes:
        fig.add_trace(
            go.Scattergl(
                x=[node["x"] for node in centroid_nodes],
                y=[node["y"] for node in centroid_nodes],
                mode="markers",
                name="Cluster centroid",
                marker=dict(
                    symbol="diamond",
                    size=9,
                    color="rgba(107, 114, 128, 0.62)",
                    line=dict(width=0.75, color="white"),
                ),
                customdata=[
                    [
                        node.get("cluster", node.get("key", "")),
                        node.get("title", node["label"]),
                    ]
                    for node in centroid_nodes
                ],
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "cluster %{customdata[0]}<extra></extra>"
                ),
                hoverinfo="skip",
            )
        )

    edges_by_type: dict[str, list[dict]] = {}
    for edge in graph.get("edges", []):
        edge_type = edge.get("type")
        if edge_type in CONNECTION_COLORS:
            edges_by_type.setdefault(edge_type, []).append(edge)

    for edge_type, edges in edges_by_type.items():
        x_values: list[float | None] = []
        y_values: list[float | None] = []
        customdata: list[list[str] | None] = []
        for edge in edges:
            source = paper_nodes_by_id.get(edge.get("source"))
            target = paper_nodes_by_id.get(edge.get("target"))
            if source is None or target is None:
                continue
            data = [
                str(edge.get("name") or edge_type.title()),
                str(edge.get("connection_types") or edge_type.title()),
                _wrap_hover_text(edge.get("description") or edge.get("rationale") or ""),
                _wrap_hover_text(edge.get("source_title") or ""),
                _wrap_hover_text(edge.get("target_title") or ""),
            ]
            edge_x = [float(source["x"])]
            edge_y = [float(source["y"])]
            edge_custom: list[list[str] | None] = [data]
            for step in range(1, EDGE_HOVER_MARKER_STEPS + 1):
                fraction = step / (EDGE_HOVER_MARKER_STEPS + 1)
                edge_x.append(
                    float(source["x"])
                    + (float(target["x"]) - float(source["x"])) * fraction
                )
                edge_y.append(
                    float(source["y"])
                    + (float(target["y"]) - float(source["y"])) * fraction
                )
                edge_custom.append(data)
            edge_x.extend([float(target["x"]), None])
            edge_y.extend([float(target["y"]), None])
            edge_custom.extend([data, None])
            x_values.extend(edge_x)
            y_values.extend(edge_y)
            customdata.extend(edge_custom)
        if x_values:
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines+markers",
                    name=edge_type.title(),
                    customdata=customdata,
                    line=dict(width=3, color=CONNECTION_COLORS[edge_type]),
                    marker=dict(size=12, color="rgba(255,255,255,0.01)"),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "type: %{customdata[1]}<br>"
                        "papers: %{customdata[3]} to %{customdata[4]}<br><br>"
                        "%{customdata[2]}<extra></extra>"
                    ),
                )
            )

    for edge_type, color in CONNECTION_COLORS.items():
        if edge_type in edges_by_type:
            continue
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=edge_type.title(),
                line=dict(width=3, color=color),
                hoverinfo="skip",
            )
        )

    if paper_nodes:
        fig.add_trace(
            go.Scatter(
                x=[node["x"] for node in paper_nodes],
                y=[node["y"] for node in paper_nodes],
                mode="markers+text",
                name="Workspace paper",
                text=[node["label"] for node in paper_nodes],
                textposition="middle center",
                textfont=dict(color="white", size=13, family="Arial Black, Arial"),
                marker=dict(
                    symbol="circle",
                    size=22,
                    color="#2563eb",
                    line=dict(width=1.75, color="white"),
                ),
                customdata=[
                    [
                        node.get("title", node["label"]),
                        node.get("key", ""),
                        node.get("primary_category", "unknown"),
                        node.get("cluster", "unknown"),
                    ]
                    for node in paper_nodes
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "arXiv: %{customdata[1]}<br>"
                    "category: %{customdata[2]}<br>"
                    "cluster: %{customdata[3]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text="Workspace Papers and Cluster Centroids",
            y=0.98,
            yanchor="top",
        ),
        template="plotly_white",
        height=760,
        margin=dict(l=18, r=18, t=72, b=18),
        hovermode="closest",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1.0,
        ),
    )
    fig.update_xaxes(visible=False, zeroline=False)
    fig.update_yaxes(visible=False, zeroline=False, scaleanchor="x", scaleratio=1)
    return fig
