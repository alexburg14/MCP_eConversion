"""BM25 search and exhaustive metadata listing."""
import json

import server
from search import search


def test_search_returns_five_ranked_results():
    results = search("perovskite", server.papers, server.index)
    assert len(results) == 5
    for r in results:
        assert r.get("doi") and r.get("title")
        # which index fired must be reported — it drives UI and eval
        assert r["matched_on"] in ("title", "abstract")


def test_search_is_deterministic():
    a = search("hydrogen evolution", server.papers, server.index)
    b = search("hydrogen evolution", server.papers, server.index)
    assert [r["doi"] for r in a] == [r["doi"] for r in b]


def test_list_papers_requires_a_filter():
    # No filter must error rather than dumping the whole corpus.
    assert "error" in json.loads(server.list_papers())


def test_list_papers_filters_by_author_accent_insensitive():
    # "Cortes" (unaccented) must match the stored "Cortés" — accent-insensitive
    # author matching is a documented guarantee.
    out = json.loads(server.list_papers(author="Cortes"))
    assert out["total_matches"] >= 1
    assert out["filters"]["author"] == "Cortes"
