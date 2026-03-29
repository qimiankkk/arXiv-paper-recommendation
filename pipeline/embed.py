"""SPECTER2 embedding wrapper for arXiv papers.

Uses the transformers + adapters library with the SPECTER2 base model and
the proximity adapter (optimized for nearest-neighbor retrieval).

Input format: title + [SEP] + abstract (max 512 tokens).
Embedding source: [CLS] token from last_hidden_state.
All embeddings are L2-normalized to unit length so dot product = cosine similarity.
"""

from __future__ import annotations

import torch
import numpy as np
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from tqdm import tqdm


class EmbeddingModel:
    """Encodes paper titles+abstracts into 768-dim SPECTER2 embeddings.

    Uses the proximity adapter for retrieval-optimized representations.

    Usage:
        model = EmbeddingModel()
        embeddings = model.embed_papers([{"title": "...", "abstract": "..."}, ...])
    """

    MODEL_NAME = "allenai/specter2_base"
    ADAPTER_NAME = "allenai/specter2"

    def __init__(self) -> None:
        """Load the SPECTER2 base model with the proximity adapter.

        Sets device to "cuda" if a GPU is available, otherwise "cpu".
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoAdapterModel.from_pretrained(self.MODEL_NAME)
        self.model.load_adapter(
            self.ADAPTER_NAME, source="hf", load_as="proximity", set_active=True
        )
        self.model.to(self.device)
        self.model.eval()

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of pre-formatted text strings into normalized embeddings.

        Args:
            texts: List of text strings (already formatted as title[SEP]abstract).

        Returns:
            np.ndarray of shape (len(texts), 768), dtype float32.
            Every row is unit-norm.
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model(**inputs)

        # [CLS] token embedding
        cls_emb = output.last_hidden_state[:, 0, :]

        # L2 normalize to unit length
        cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=1)

        return cls_emb.cpu().numpy().astype(np.float32)

    def embed_papers(self, papers: list[dict]) -> np.ndarray:
        """Encode paper dicts into normalized embeddings.

        Each paper dict must have "title" and "abstract" keys.
        Input format: title + [SEP] + abstract (per SPECTER2 spec).

        Args:
            papers: List of dicts, each with at least "title" and "abstract".

        Returns:
            np.ndarray of shape (len(papers), 768), dtype float32.
        """
        sep = self.tokenizer.sep_token
        texts = [
            p["title"] + sep + (p.get("abstract") or "")
            for p in papers
        ]

        # Process in batches of 64
        batch_size = 64
        all_embeddings: list[np.ndarray] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i : i + batch_size]
            emb = self.embed_batch(batch)
            all_embeddings.append(emb)

        return np.concatenate(all_embeddings, axis=0)
