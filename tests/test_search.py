"""BM25 search and exhaustive metadata listing."""
import pytest

pytestmark = pytest.mark.integration  # imports the real caches
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


def test_list_papers_without_a_filter_lists_the_corpus_within_the_limit():
    # Listing everything is bounded by the limit anyway; refusing it only sent
    # the model in circles ("collaboration over time" asked eight times).
    out = json.loads(server.list_papers(limit=3))
    assert out["total_matches"] == len(server.papers)
    assert out["returned"] == len(out["papers"]) == 3
    years = [int(p["year"]) for p in out["papers"] if str(p["year"]).isdigit()]
    assert years == sorted(years, reverse=True)


def test_list_papers_filters_by_author_accent_insensitive():
    # "Cortes" (unaccented) must match the stored "Cortés" — accent-insensitive
    # author matching is a documented guarantee.
    out = json.loads(server.list_papers(author="Cortes"))
    assert out["total_matches"] >= 1
    assert out["filters"]["author"] == "Cortes"
