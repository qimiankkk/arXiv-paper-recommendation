"""
Embedding artifact and runtime diagnostics for onboarding.

To run the test:
    pytest -s -m embedding tests/test_embedding_diagnostics.py
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from pipeline.concept_tags import (
    CONCEPT_EMBEDDINGS_FILE,
    CONCEPT_EMBEDDINGS_META_FILE,
    CONCEPT_TAG_MAP,
    load_concept_embedding_artifacts,
)
from pipeline.interest_expander import expand_interest
from tests.test_profile_v2_pipeline import (
    DATA_DIR,
    _unit_rows,
    assert_unit_norm_rows,
)


def _concept_matrix() -> tuple[list[str], list[str], np.ndarray, Path]:
    path = DATA_DIR / CONCEPT_EMBEDDINGS_FILE
    meta_path = DATA_DIR / CONCEPT_EMBEDDINGS_META_FILE
    if not path.exists() or not meta_path.exists():
        pytest.skip(f"Missing concept embedding artifact(s): {path}, {meta_path}")
    embeddings = load_concept_embedding_artifacts(DATA_DIR)
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    keys = meta["keys"]
    labels = [CONCEPT_TAG_MAP[key].label if key in CONCEPT_TAG_MAP else key for key in keys]
    matrix = np.stack([embeddings[key] for key in keys]).astype(np.float32)
    return keys, labels, matrix, path


def _canonical(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _find_concept(candidates: list[str], keys: list[str], labels: list[str]) -> int | None:
    lookup = {
        _canonical(key): i
        for i, key in enumerate(keys)
    }
    lookup.update({_canonical(label): i for i, label in enumerate(labels)})
    for candidate in candidates:
        idx = lookup.get(_canonical(candidate))
        if idx is not None:
            return idx
    for candidate in candidates:
        c = _canonical(candidate)
        for i, (key, label) in enumerate(zip(keys, labels)):
            if c and (c in _canonical(key) or c in _canonical(label)):
                return i
    return None


def _top_neighbors(
    query: np.ndarray,
    matrix: np.ndarray,
    labels: list[str],
    keys: list[str],
    top_k: int = 5,
) -> list[tuple[int, str, str, float]]:
    sims = np.asarray(query, dtype=np.float32) @ matrix.T
    order = np.argsort(-sims)[:top_k]
    return [(int(i), keys[int(i)], labels[int(i)], float(sims[int(i)])) for i in order]


def _print_neighbors(title: str, neighbors: list[tuple[int, str, str, float]]) -> None:
    print(title)
    for rank, (_, key, label, sim) in enumerate(neighbors, start=1):
        print(f"  {rank:>2}. {label} ({key}) sim={sim:.4f}")


@lru_cache(maxsize=1)
def _require_real_embedding_model():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from pipeline.embed import EmbeddingModel

        return EmbeddingModel()
    except Exception as exc:
        pytest.skip(
            "Real SPECTER2 embedding model is unavailable locally/offline: "
            f"{type(exc).__name__}: {exc}"
        )


def _embed_texts_with_real_model(texts: list[str], model: object | None = None) -> np.ndarray:
    if model is None:
        model = _require_real_embedding_model()
    papers = [{"title": text, "abstract": ""} for text in texts]
    return _unit_rows(model.embed_papers(papers))


@pytest.mark.slow
@pytest.mark.embedding
@pytest.mark.diagnostic
def test_free_text_embedding_runtime_is_reasonable():
    free_texts = [
        "single-cell perturbation modeling",
        "diffusion models for medical imaging",
        "reinforcement learning for robot navigation",
        "neural operators for weather forecasting",
        "LLMs for clinical decision support",
    ]
    expanded = [expand_interest(text) for text in free_texts]
    model = _require_real_embedding_model()
    start = time.perf_counter()
    matrix = _embed_texts_with_real_model(expanded, model)
    elapsed = time.perf_counter() - start

    print("\nFREE-TEXT RUNTIME EMBEDDING")
    print(f"text count: {len(free_texts)}")
    print(f"total seconds: {elapsed:.3f}")
    print(f"average seconds: {elapsed / len(free_texts):.3f}")
    for raw, exp in zip(free_texts, expanded):
        print(f"raw={raw!r}")
        print(f"expanded={exp!r}")

    assert matrix.shape == (len(free_texts), matrix.shape[1])
    assert matrix.shape[1] > 0
    assert np.isfinite(matrix).all()
    assert_unit_norm_rows(matrix)
    assert elapsed < float(os.getenv("FREE_TEXT_EMBEDDING_MAX_SECONDS", "20"))


@pytest.mark.slow
@pytest.mark.embedding
@pytest.mark.diagnostic
def test_free_text_embeddings_have_reasonable_nearest_concepts():
    keys, labels, concept_matrix, _ = _concept_matrix()
    queries = [
        (
            "single-cell perturbation modeling",
            ["Computational Biology", "Bioinformatics", "Machine Learning", "Healthcare AI"],
            [],
        ),
        (
            "diffusion models for medical imaging",
            ["Medical Imaging", "Healthcare AI", "Computer Vision", "Machine Learning"],
            ["Number Theory", "Astrophysics"],
        ),
        (
            "neural operators for weather forecasting",
            ["Climate & Weather", "Scientific Machine Learning", "Machine Learning"],
            [],
        ),
        (
            "reinforcement learning for robot navigation",
            ["Robotics & Embodied AI", "Robotics", "Reinforcement Learning", "Machine Learning"],
            [],
        ),
    ]

    expanded = [expand_interest(query) for query, _, _ in queries]
    query_matrix = _embed_texts_with_real_model(expanded)

    for row, (query, expected, forbidden) in zip(query_matrix, queries):
        expected_indices = {
            idx
            for name in expected
            if (idx := _find_concept([name], keys, labels)) is not None
        }
        if not expected_indices:
            print(f"Skipping query expectation for {query!r}; no expected concepts exist")
            continue
        neighbors = _top_neighbors(row, concept_matrix, labels, keys, top_k=5)
        _print_neighbors(f"\nquery={query!r}", neighbors)
        top_indices = {idx for idx, _, _, _ in neighbors}
        assert expected_indices & top_indices
        for forbidden_name in forbidden:
            forbidden_idx = _find_concept([forbidden_name], keys, labels)
            if forbidden_idx is not None:
                assert neighbors[0][0] != forbidden_idx


@pytest.mark.slow
@pytest.mark.embedding
@pytest.mark.diagnostic
def test_free_text_expansion_improves_or_preserves_neighbor_quality():
    keys, labels, concept_matrix, _ = _concept_matrix()
    cases = [
        ("healthcare", ["Healthcare AI"]),
        ("climate", ["Climate & Weather"]),
        ("computational biology", ["Computational Biology"]),
    ]

    raw_texts = [phrase for phrase, _ in cases]
    expanded_texts = [expand_interest(phrase) for phrase, _ in cases]
    raw_matrix = _embed_texts_with_real_model(raw_texts)
    expanded_matrix = _embed_texts_with_real_model(expanded_texts)
    assert_unit_norm_rows(raw_matrix)
    assert_unit_norm_rows(expanded_matrix)

    for i, (phrase, intended_names) in enumerate(cases):
        intended_idx = _find_concept(intended_names, keys, labels)
        if intended_idx is None:
            print(f"Skipping {phrase!r}; intended concept {intended_names} is absent")
            continue
        raw_neighbors = _top_neighbors(raw_matrix[i], concept_matrix, labels, keys, top_k=5)
        expanded_neighbors = _top_neighbors(expanded_matrix[i], concept_matrix, labels, keys, top_k=5)
        raw_order = np.argsort(-(raw_matrix[i] @ concept_matrix.T)).tolist()
        expanded_order = np.argsort(-(expanded_matrix[i] @ concept_matrix.T)).tolist()
        raw_rank = raw_order.index(intended_idx) + 1
        expanded_rank = expanded_order.index(intended_idx) + 1

        _print_neighbors(f"\nraw phrase={phrase!r}", raw_neighbors)
        _print_neighbors(f"expanded phrase={expanded_texts[i]!r}", expanded_neighbors)
        print(f"intended concept rank raw={raw_rank} expanded={expanded_rank}")
        assert intended_idx in {idx for idx, _, _, _ in expanded_neighbors}


@pytest.mark.diagnostic
def test_concept_embedding_artifact_loads_and_matches_registry():
    keys, labels, matrix, path = _concept_matrix()
    norms = np.linalg.norm(matrix, axis=1)

    print("\nCONCEPT EMBEDDING ARTIFACT")
    print(f"artifact path: {path}")
    print(f"number of concepts: {len(keys)}")
    print(f"embedding_dim: {matrix.shape[1]}")
    print(f"norm min/mean/max: {norms.min():.6f}/{norms.mean():.6f}/{norms.max():.6f}")

    assert matrix.ndim == 2
    assert matrix.shape[1] > 0
    assert matrix.shape[0] == len(keys)
    assert matrix.shape[0] == len([key for key in keys if key in CONCEPT_TAG_MAP])
    assert np.isfinite(matrix).all()
    assert_unit_norm_rows(matrix)
    assert np.all(norms > 0)
    D = 1.0 - matrix @ matrix.T
    non_self = D[~np.eye(D.shape[0], dtype=bool)]
    assert float(non_self.min()) > 1e-6, "Unexpected duplicate concept vectors"
    assert labels


@pytest.mark.diagnostic
def test_concept_embedding_nearest_neighbors_are_semantically_reasonable():
    keys, labels, matrix, _ = _concept_matrix()
    anchors = {
        "healthcare_ai": ["medical_imaging", "computational_biology", "llms"],
        "climate_weather": ["scientific_ml", "time_series"],
        "llms": ["ai_for_code", "ai_safety", "speech_audio"],
        "robotics_embodied": ["reinforcement_learning", "autonomous_driving"],
        "medical_imaging": ["healthcare_ai", "computer_vision", "generative_models"],
        "computational_biology": ["drug_discovery", "healthcare_ai"],
        "scientific_ml": ["climate_weather", "quantum_computing"],
    }

    for anchor, expected in anchors.items():
        anchor_idx = _find_concept([anchor], keys, labels)
        if anchor_idx is None:
            print(f"Skipping missing anchor {anchor}")
            continue
        neighbors = [
            item
            for item in _top_neighbors(matrix[anchor_idx], matrix, labels, keys, top_k=11)
            if item[0] != anchor_idx
        ][:10]
        _print_neighbors(f"\nanchor={labels[anchor_idx]} ({anchor})", neighbors)
        expected_indices = {
            idx
            for name in expected
            if (idx := _find_concept([name], keys, labels)) is not None
        }
        if expected_indices:
            assert expected_indices & {idx for idx, _, _, _ in neighbors[:5]}

    triples = [
        ("healthcare_ai", "medical_imaging", "quantum_computing"),
        ("climate_weather", "scientific_ml", "drug_discovery"),
        ("llms", "ai_for_code", "robotics_embodied"),
    ]
    D = 1.0 - matrix @ matrix.T
    for anchor, related, unrelated in triples:
        ia = _find_concept([anchor], keys, labels)
        ir = _find_concept([related], keys, labels)
        iu = _find_concept([unrelated], keys, labels)
        if None in (ia, ir, iu):
            print(f"Skipping triple: {anchor}, {related}, {unrelated}")
            continue
        print(
            f"triple {anchor}: dist({related})={D[ia, ir]:.4f} "
            f"dist({unrelated})={D[ia, iu]:.4f}"
        )
        assert D[ia, ir] < D[ia, iu]


@pytest.mark.diagnostic
def test_print_concept_pair_distances_for_threshold_tuning():
    keys, labels, matrix, _ = _concept_matrix()
    D = 1.0 - matrix @ matrix.T
    mask = ~np.eye(D.shape[0], dtype=bool)
    off_diag = D[mask]
    quantiles = np.quantile(off_diag, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])

    print("\nCONCEPT DISTANCE QUANTILES")
    print("q05 q10 q25 q50 q75 q90 q95")
    print(" ".join(f"{q:.4f}" for q in quantiles))

    pairs: list[tuple[float, int, int]] = []
    for i in range(D.shape[0]):
        for j in range(i + 1, D.shape[0]):
            pairs.append((float(D[i, j]), i, j))

    print("\nCLOSEST 20 CONCEPT PAIRS")
    for dist, i, j in sorted(pairs)[:20]:
        print(f"{dist:.4f} | {labels[i]} ({keys[i]}) <-> {labels[j]} ({keys[j]})")

    print("\nFARTHEST 20 CONCEPT PAIRS")
    for dist, i, j in sorted(pairs, reverse=True)[:20]:
        print(f"{dist:.4f} | {labels[i]} ({keys[i]}) <-> {labels[j]} ({keys[j]})")

    curated_pairs = [
        ("machine_learning", "deep_learning"),
        ("machine_learning", "healthcare_ai"),
        ("healthcare_ai", "medical_imaging"),
        ("natural_language_processing", "llms"),
        ("computer_vision", "medical_imaging"),
        ("healthcare_ai", "astrophysics"),
        ("robotics_embodied", "quantum_computing"),
        ("climate_weather", "scientific_ml"),
    ]
    print("\nCURATED PAIR DISTANCES")
    for left, right in curated_pairs:
        i = _find_concept([left], keys, labels)
        j = _find_concept([right], keys, labels)
        if i is None or j is None:
            print(f"skipped {left} vs {right}: missing concept")
            continue
        print(f"{left} vs {right}: {D[i, j]:.4f}")

    assert np.isfinite(off_diag).all()
    assert float(off_diag.min()) >= -1e-4
    assert float(off_diag.max()) <= 2.0 + 1e-4
    assert float(off_diag.min()) > 1e-6
