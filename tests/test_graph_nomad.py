"""Collaboration-graph queries and NOMAD input validation.

The NOMAD tests deliberately exercise only the no-filter validation path, which
returns before any HTTP call — the suite must not depend on the network.
"""
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
