"""Collaboration-graph queries and NOMAD input validation.

The NOMAD tests deliberately exercise only the no-filter validation path, which
returns before any HTTP call — the suite must not depend on the network.
"""
import pytest

pytestmark = pytest.mark.integration  # imports the real caches
import graph
import nomad_search


def test_get_collaborators_resolves_known_pi():
    r = graph.get_collaborators("Sharp")
    assert "error" not in r
    assert r["collaborator_count"] >= 1
    assert isinstance(r["collaborators"], list)


def test_joint_papers_unknown_pi_errors_not_crashes():
    r = graph.joint_papers("Sharp", "ZZ_definitely_not_a_pi")
    assert "error" in r


def test_search_nomad_requires_a_filter_without_network():
    # All filters empty -> validation error, no request issued.
    out = nomad_search.search_nomad()
    assert "error" in out


def test_search_nomad_strips_titles_from_author():
    # Regression guard for the title-prefix bug: the cleaned author is echoed
    # back in the filters. Uses the validation echo, still no network call when
    # paired with the empty-filter guard — so assert on _strip_titles directly.
    assert nomad_search._strip_titles("Prof. Dr. Karsten Reuter") == "Karsten Reuter"


def test_graph_edges_are_deduplicated_against_the_brace_quirk():
    G = graph._load()
    for _, _, data in G.edges(data=True):
        dois = data.get("shared_dois", [])
        assert len(dois) == len(set(dois)) == data["weight"]
        assert not any(d.endswith("}") for d in dois)


def test_centrality_ranks_by_the_asked_measure():
    by_partners = graph.collaboration_centrality(5, by="collaborators")
    counts = [r["collaborators"] for r in by_partners]
    assert counts == sorted(counts, reverse=True)
    assert {"betweenness", "collaborators", "shared_papers"} <= set(by_partners[0])
    assert graph.collaboration_centrality(1, by="nonsense")[0]["betweenness"] >= 0


def test_joint_papers_carry_titles_through_the_tool():
    import json

    import server

    out = json.loads(server.joint_papers("Eichhorn", "Sharp"))
    assert out["shared_total"] == out["in_corpus"] + len(out["not_in_corpus"])
    assert out["in_corpus"] == len(out["papers"])
    assert out["papers"] and out["papers"][0]["title"]
