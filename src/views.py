"""Data behind the non-chat pages (corpus map, collaboration graph).

Pure data producers: the frontend renders them with deck.gl / d3. Both are
memoised per process because UMAP is ~30 s cold; a lock keeps two first
requests from running it twice.
"""
from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path

import corpus_map
import semantic_search

_REPO_ROOT = Path(__file__).resolve().parent.parent
COLLAB_GRAPH_PATH = _REPO_ROOT / "data" / "cache" / "collaboration_graph.json"

# Corpus-map cluster colors: Tableau-20, saturated hues first so the common
# small-cluster-count case gets the most distinguishable set. RGB triples.
CLUSTER_PALETTE = [
    tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
    for h in (
        "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
        "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#79706E",
        "#9ECAE9", "#FFBF79", "#88D27A", "#FF9D98", "#83BCB6",
        "#F2CF5B", "#D6A5C9", "#D67195", "#D8B5A5", "#BAB0AC",
    )
]

MIN_CLUSTERS, MAX_CLUSTERS, DEFAULT_CLUSTERS = 2, 20, 8

_map_lock = threading.Lock()


def corpus_map_available() -> bool:
    return corpus_map.is_available()


@lru_cache(maxsize=8)
def _corpus_map_cached(n_clusters: int) -> dict:
    import server  # local import: the caches are loaded by the entry point

    rows = corpus_map.build_map(server.papers_by_doi, n_clusters=n_clusters)
    clusters = sorted({r["cluster"] for r in rows})
    cmap = {c: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i, c in enumerate(clusters)}
    # deck.gl's OrthographicView has y pointing down, so negate y for a
    # conventional (y-up) orientation.
    points = [
        {
            "doi": r["doi"], "x": float(r["x"]), "y": -float(r["y"]),
            "title": r["title"], "year": str(r["year"]),
            "cluster": r["cluster"], "color": list(cmap[r["cluster"]]),
        }
        for r in rows
    ]
    legend = [{"cluster": c, "color": list(rgb)} for c, rgb in cmap.items()]
    return {"points": points, "legend": legend}


def corpus_map_payload(n_clusters: int = DEFAULT_CLUSTERS) -> dict:
    """UMAP points + legend for ``n_clusters`` (clamped to the slider range)."""
    n_clusters = max(MIN_CLUSTERS, min(int(n_clusters), MAX_CLUSTERS))
    with _map_lock:
        return _corpus_map_cached(n_clusters)


@lru_cache(maxsize=1)
def collab_graph_payload() -> dict | None:
    """PI co-authorship graph shaped for the d3 force map: nodes with degree +
    a short surname label, and undirected weighted links. None if not built."""
    if not COLLAB_GRAPH_PATH.exists():
        return None
    g = json.loads(COLLAB_GRAPH_PATH.read_text(encoding="utf-8"))
    deg: dict[str, int] = {}
    for link in g["links"]:
        deg[link["source"]] = deg.get(link["source"], 0) + 1
        deg[link["target"]] = deg.get(link["target"], 0) + 1
    nodes = [
        {
            "id": n["id"], "name": n["name"], "label": (n["name"].split() or [n["name"]])[-1],
            "group": n.get("group", ""), "inst": n.get("institution", ""),
            "papers": n.get("paper_count", 0), "deg": deg.get(n["id"], 0),
        }
        for n in g["nodes"]
    ]
    links = [{"source": l["source"], "target": l["target"], "weight": l["weight"]} for l in g["links"]]
    return {"nodes": nodes, "links": links}


def text_similarity_available() -> bool:
    return semantic_search.is_available()


def text_similarity_payload(text: str, top_k: int = 10) -> list[dict]:
    """Papers closest to an arbitrary pasted abstract, by cosine similarity in
    the same BGE-small embedding space as the publication map — a query-to-doc
    search (see semantic_search.semantic_search), not the map's 2D layout."""
    import server  # local import: the caches are loaded by the entry point

    results = semantic_search.semantic_search(text, server.papers_by_doi, top_k=top_k)
    return [
        {"doi": r.get("doi", ""), "title": r.get("title", "(untitled)"),
         "year": str(r.get("year", "")), "score": round(float(r.get("semantic_score", 0.0)), 4)}
        for r in results
    ]
