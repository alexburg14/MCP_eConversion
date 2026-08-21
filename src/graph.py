"""Queries over the PI co-authorship graph (collaboration_graph.json).

Answers the network questions BM25/semantic search cannot: who collaborates with
whom, which group bridges otherwise-separate groups, and which clusters of groups
work together. All deterministic graph analytics — no model in the loop.

The graph is lazy-loaded on first call so importing this module stays cheap.
Build the cache with: python src/scripts/build_graph_cache.py
"""
import json
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_GRAPH_PATH = _DATA_DIR / "cache" / "collaboration_graph.json"

_graph: nx.Graph | None = None


def is_available() -> bool:
    return _GRAPH_PATH.exists()


def _load() -> nx.Graph:
    """Idempotent lazy load. Raises if the cache is missing."""
    global _graph
    if _graph is not None:
        return _graph
    if not _GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"{_GRAPH_PATH.name} not found. Run: python src/scripts/build_graph_cache.py"
        )
    with open(_GRAPH_PATH, encoding="utf-8") as f:
        _graph = json_graph.node_link_graph(json.load(f), edges="links")
    return _graph


def _resolve(query: str, G: nx.Graph) -> str | None:
    """Map a free-text PI query to a node id (smid).

    Exact smid first, then case-insensitive substring on name / last name / group.
    On multiple matches, prefer the most-published PI so the obvious person wins.
    """
    if query in G:
        return query
    q = query.strip().lower()
    matches = [
        n for n, d in G.nodes(data=True)
        if q in d["name"].lower() or q in d.get("group", "").lower()
    ]
    if not matches:
        return None
    return max(matches, key=lambda n: G.nodes[n]["paper_count"])


def _node(G: nx.Graph, n: str) -> dict:
    d = G.nodes[n]
    return {"name": d["name"], "group": d.get("group", ""), "institution": d.get("institution", "")}


def get_collaborators(pi_query: str) -> dict:
    """All PIs who share at least one paper with the queried PI, most-shared first."""
    G = _load()
    n = _resolve(pi_query, G)
    if n is None:
        return {"error": f"No PI matching {pi_query!r}."}
    collaborators = [
        {**_node(G, other), "shared_papers": G.edges[n, other]["weight"]}
        for other in G.neighbors(n)
    ]
    collaborators.sort(key=lambda c: -c["shared_papers"])
    return {**_node(G, n), "collaborator_count": len(collaborators), "collaborators": collaborators}


def joint_papers(pi_a: str, pi_b: str) -> dict:
    """Shared-paper DOIs between two PIs (empty if they never co-published)."""
    G = _load()
    na, nb = _resolve(pi_a, G), _resolve(pi_b, G)
    if na is None:
        return {"error": f"No PI matching {pi_a!r}."}
    if nb is None:
        return {"error": f"No PI matching {pi_b!r}."}
    if not G.has_edge(na, nb):
        return {"pi_a": _node(G, na), "pi_b": _node(G, nb), "count": 0, "dois": []}
    e = G.edges[na, nb]
    return {"pi_a": _node(G, na), "pi_b": _node(G, nb), "count": e["weight"], "dois": e["shared_dois"]}


def collaboration_centrality(top_k: int = 10) -> list[dict]:
    """PIs ranked by betweenness centrality — the 'bridges' between groups.

    Unweighted: a high score means a PI sits on many shortest paths between
    otherwise weakly-connected groups, not simply that they publish a lot.
    """
    G = _load()
    bc = nx.betweenness_centrality(G)
    ranked = sorted(bc.items(), key=lambda kv: -kv[1])[:top_k]
    return [{**_node(G, n), "betweenness": round(score, 4)} for n, score in ranked]


def collaboration_communities() -> dict:
    """Greedy-modularity clusters of PIs who collaborate internally.

    Singletons (PIs with no shared papers) are reported as a count, not listed.
    """
    G = _load()
    communities = nx.community.greedy_modularity_communities(G)
    groups, singletons = [], 0
    for c in communities:
        if len(c) == 1:
            singletons += 1
            continue
        members = sorted((_node(G, n)["name"] for n in c))
        groups.append({"size": len(members), "members": members})
    groups.sort(key=lambda g: -g["size"])
    return {"community_count": len(groups), "unconnected_pis": singletons, "communities": groups}


if __name__ == "__main__":
    # Smoke test / verification against known-good values from the cached graph.
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # PI names carry umlauts
    print("available:", is_available())

    print("\n== get_collaborators('Sharp') ==")
    r = get_collaborators("Sharp")
    print(r["name"], "->", r["collaborator_count"], "collaborators")
    for c in r["collaborators"][:5]:
        print(f"  {c['shared_papers']:3d}  {c['name']}")

    print("\n== joint_papers('Eichhorn', 'Sharp') ==")
    j = joint_papers("Eichhorn", "Sharp")
    print(f"  {j['count']} shared papers")

    print("\n== collaboration_centrality(top 5) ==")
    for c in collaboration_centrality(5):
        print(f"  {c['betweenness']:.4f}  {c['name']}")

    print("\n== collaboration_communities() ==")
    cc = collaboration_communities()
    print(f"  {cc['community_count']} communities, {cc['unconnected_pis']} unconnected PIs")
    for g in cc["communities"]:
        print(f"  size {g['size']}: {', '.join(m.split('Dr. ')[-1] for m in g['members'])}")
