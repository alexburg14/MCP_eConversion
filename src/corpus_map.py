"""UMAP + KMeans map of the embeddings cache.

Visual answer to the 2026-07-14 meeting note's "which papers are near the
one I'm reading? -> clustering by keywords or latent space representation".

Division of labor:
- KMeans clusters in the FULL 384-d embedding space — the same geometry the
  semantic-search tools rank by. Clustering the 2D projection instead would
  group projection artifacts (the 2D plane keeps ~14% of the variance).
- UMAP produces the 2D layout for display only. Unlike PCA it preserves
  local neighborhoods, so semantically tight groups render as visually
  tight islands. cosine metric to match how the vectors are compared.
- Clusters are labeled with their top TF-IDF terms over member titles, so
  the legend reads "perovskite / solar / cells" instead of "cluster 3".

UMAP is expensive (~30s cold, numba JIT included); _project() is memoized
per process. KMeans per cluster-count is memoized too, so moving the
cluster slider never re-runs UMAP.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_CACHE_PATH = _DATA_DIR / "cache" / "embeddings_cache.npz"

_TERMS_PER_LABEL = 3


def is_available() -> bool:
    return _CACHE_PATH.exists()


@lru_cache(maxsize=1)
def _load() -> tuple[tuple[str, ...], np.ndarray]:
    npz = np.load(_CACHE_PATH, allow_pickle=True)
    dois = tuple(str(d).lower() for d in npz["dois"])
    return dois, npz["vectors"]


@lru_cache(maxsize=1)
def _project() -> np.ndarray:
    import umap  # deferred: heavy import (numba), only needed for the map tab

    _, vectors = _load()
    reducer = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1,
        metric="cosine", random_state=42,
    )
    return reducer.fit_transform(vectors)


@lru_cache(maxsize=8)
def _cluster(n_clusters: int) -> np.ndarray:
    _, vectors = _load()
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(vectors)


def _label_clusters(titles: list[str], assignments: np.ndarray, n_clusters: int) -> dict[int, str]:
    """Top TF-IDF terms of each cluster's concatenated titles, joined with ' / '.

    TF-IDF (not raw counts) so corpus-wide fillers like 'properties' don't
    label every cluster; sublinear_tf damps any single verbose title.
    """
    docs = ["" for _ in range(n_clusters)]
    for title, c in zip(titles, assignments):
        docs[c] += " " + title
    tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=1)
    matrix = tfidf.fit_transform(docs)
    vocab = np.array(tfidf.get_feature_names_out())
    labels = {}
    for c in range(n_clusters):
        row = matrix[c].toarray().ravel()
        top = vocab[np.argsort(-row)[:_TERMS_PER_LABEL]]
        labels[c] = " / ".join(top)
    return labels


def build_map(papers_by_doi: dict, n_clusters: int = 8) -> list[dict]:
    """One row per cached paper: doi, title, year, x/y (UMAP), keyword-labeled cluster."""
    dois, _ = _load()
    coords = _project()
    n_clusters = max(1, min(n_clusters, len(dois)))
    assignments = _cluster(n_clusters)

    titles = [str(papers_by_doi.get(d, {}).get("title") or "") for d in dois]
    labels = _label_clusters(titles, assignments, n_clusters)

    rows = []
    for i, doi in enumerate(dois):
        paper = papers_by_doi.get(doi, {})
        rows.append({
            "doi": doi,
            "title": paper.get("title") or doi,
            "year": paper.get("year") or "",
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "cluster": labels[assignments[i]],
        })
    return rows
