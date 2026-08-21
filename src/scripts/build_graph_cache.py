"""Build collaboration_graph.json — the PI co-authorship graph.

Two PIs are linked iff they share at least one publication DOI. Because each PI
in pis_cache.json carries its own `publication_dois`, every edge is factual set
intersection — no author-name disambiguation. The graph is tiny (~42 nodes), so
it is cached as node-link JSON and loaded into networkx at runtime by graph.py.

Re-run only when pis_cache.json changes.
"""
import json
from itertools import combinations
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PIS = DATA_DIR / "cache" / "pis_cache.json"
OUTPUT = DATA_DIR / "cache" / "collaboration_graph.json"


def _doi_set(pi: dict) -> set:
    return {d.strip().lower() for d in (pi.get("publication_dois") or []) if d and d.strip()}


def main():
    if not PIS.exists():
        raise SystemExit(f"Missing {PIS.name}. Run build_pis_cache.py first.")
    with open(PIS, encoding="utf-8") as f:
        pis = json.load(f)

    G = nx.Graph(built_from="pis_cache.json", n_pis=len(pis))

    # Every PI is a node — even the 8 with no recorded papers — so lookups always
    # resolve; those just end up isolated (degree 0).
    dois_by_pi = {}
    for pi in pis:
        smid = pi["smid"]
        dois = _doi_set(pi)
        dois_by_pi[smid] = dois
        G.add_node(
            smid,
            name=pi["name"],
            group=pi.get("group", ""),
            institution=pi.get("institution", ""),
            paper_count=len(dois),
        )

    # Edge weight = number of shared papers; keep the shared DOIs for joint_papers().
    for a, b in combinations(dois_by_pi, 2):
        shared = dois_by_pi[a] & dois_by_pi[b]
        if shared:
            G.add_edge(a, b, weight=len(shared), shared_dois=sorted(shared))

    data = json_graph.node_link_data(G, edges="links")
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Wrote {OUTPUT.name}: {G.number_of_nodes()} PIs, {G.number_of_edges()} edges")
    isolated = [n for n, d in G.degree() if d == 0]
    print(f"  {len(isolated)} PIs with no recorded collaborations")


if __name__ == "__main__":
    main()
