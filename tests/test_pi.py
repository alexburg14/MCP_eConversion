"""PI name normalization and lookup — the parts that have silently broken before."""
import json

import pytest

pytestmark = pytest.mark.integration  # imports the real caches

import server
from nomad_search import _strip_titles


@pytest.mark.parametrize("raw,expected", [
    ("Müller", "muller"),      # umlaut folds
    ("Cortés", "cortes"),      # accent folds
    ("Rinke", "rinke"),        # plain lowercases
])
def test_fold_strips_accents(raw, expected):
    assert server._fold(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("Prof. Dr. Karsten Reuter", "Karsten Reuter"),  # the bug: titles blocked NOMAD matches
    ("Dr. Patrick Rinke", "Patrick Rinke"),
    ("Patrick Rinke", "Patrick Rinke"),              # no titles: unchanged
])
def test_strip_titles(raw, expected):
    assert _strip_titles(raw) == expected


def test_query_tokens_drops_stopwords_and_short_words():
    # "the" is a stopword, "of"/"a" are too short — none should survive as signal.
    assert server._query_tokens("the energy of a conversion") == ["energy", "conversion"]


def test_get_pi_resolves_single_match():
    out = json.loads(server.get_pi("Rinke"))
    assert "error" not in out
    assert "Rinke" in out["name"]


def test_get_pi_rejects_empty():
    assert "error" in json.loads(server.get_pi(""))


def test_list_pis_returns_everyone_with_what_the_corpus_holds():
    out = json.loads(server.list_pis())
    assert out["count"] == len(out["results"]) == len(server._PIS)
    rinke = next(p for p in out["results"] if "Rinke" in p["name"])
    assert rinke["research_focus"] and "papers_in_corpus" in rinke
    assert "publications" not in rinke, "the detail is get_pi's job"


def test_most_collaborative_papers_ranks_and_counts_per_year():
    out = json.loads(server.most_collaborative_papers(limit=3))
    first = out["results"][0]
    assert first["pi_count"] == len(first["pis"]) >= 2
    assert first["title"] and "abstract" not in first
    assert sum(y["papers"] for y in out["by_year"].values()) == out["papers_with_a_pi"]


def test_count_papers_sizes_many_topics_in_one_call():
    out = json.loads(server.count_papers(["perovskite", "aardvark zebra", "a"]))
    assert out["papers"] == len(server.papers)
    assert out["counts"]["perovskite"]["title_or_abstract"] > 0
    assert out["counts"]["aardvark zebra"] == {"title_or_abstract": 0, "full_text": 0}
    assert "error" in out["counts"]["a"]


def test_pi_dois_lose_the_stray_brace():
    for pi in server._PIS:
        assert not any(d.endswith("}") for d in server._pi_dois(pi))
