"""PI name normalization and lookup — the parts that have silently broken before."""
import json

import pytest

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
