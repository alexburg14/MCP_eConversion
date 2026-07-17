"""Cosine semantic search over the BGE-small embeddings cache.

Complements BM25 in search.py — call this when the user query uses different
vocabulary than the abstracts (synonyms, paraphrases, conceptual framings).

Model + vectors are lazy-loaded on first call so the MCP server starts cheap.
"""
import json
from pathlib import Path

import numpy as np

from search import apply_cache

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_CACHE_PATH = _DATA_DIR / "embeddings_cache.npz"

_dois: np.ndarray | None = None
_vectors: np.ndarray | None = None
_doi_index: dict | None = None
_model = None


def _load():
    """Idempotent lazy load. Returns (dois, vectors, model) or raises if cache missing."""
    global _dois, _vectors, _doi_index, _model
    if _vectors is not None and _model is not None:
        return _dois, _vectors, _model
    if not _CACHE_PATH.exists():
        raise FileNotFoundError(
            f"{_CACHE_PATH.name} not found. Run: python src/build_embeddings_cache.py"
        )
    npz = np.load(_CACHE_PATH, allow_pickle=True)
    _dois = npz["dois"]
    _vectors = npz["vectors"]
    _doi_index = {str(d).lower(): i for i, d in enumerate(_dois)}
    model_name = str(npz["model"]) if "model" in npz.files else "BAAI/bge-small-en-v1.5"
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(model_name)
    return _dois, _vectors, _model


def is_available() -> bool:
    return _CACHE_PATH.exists()


def semantic_search(query: str, papers_by_doi: dict, top_k: int = 5) -> list[dict]:
    dois, vectors, model = _load()
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    scores = vectors @ q[0]  # cosine, vectors already L2-normalised at build time
    top_idx = np.argpartition(-scores, top_k)[:top_k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    results = []
    for i in top_idx:
        doi = str(dois[i]).lower()
        paper = dict(papers_by_doi.get(doi, {"doi": doi}))
        apply_cache(paper, doi)
        paper["semantic_score"] = float(scores[i])
        paper["matched_on"] = "semantic"
        results.append(paper)
    return results


def similar_papers(doi: str, papers_by_doi: dict, top_k: int = 5) -> list[dict] | None:
    """Papers whose embedding is closest to the given paper's own embedding.

    Doc-to-doc similarity (vs. semantic_search's query-to-doc) — answers
    "what else is like the paper I'm reading?" using the same cached vectors.
    Returns None if the DOI isn't in the embeddings cache.
    """
    dois, vectors, _ = _load()
    doi_key = doi.strip().lower()
    idx = _doi_index.get(doi_key)
    if idx is None:
        return None

    scores = vectors @ vectors[idx]
    scores[idx] = -1.0  # exclude the paper itself
    top_idx = np.argpartition(-scores, top_k)[:top_k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    results = []
    for i in top_idx:
        d = str(dois[i]).lower()
        paper = dict(papers_by_doi.get(d, {"doi": d}))
        apply_cache(paper, d)
        paper["similarity_score"] = float(scores[i])
        results.append(paper)
    return results
