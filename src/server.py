import json
import re
import unicodedata
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from search import load_papers, build_index, search, _ABSTRACTS, apply_cache
import semantic_search
import graph as _graph

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = _DATA_DIR / "data_publication_dois.csv"
FULLTEXT_CACHE_PATH = _DATA_DIR / "fulltext_cache.json"
PIS_CACHE_PATH = _DATA_DIR / "pis_cache.json"

mcp = FastMCP("eConversion Papers")

papers = load_papers(CSV_PATH)
index = build_index(papers)
papers_by_doi = {p["doi"].lower(): p for p in papers}

_FULLTEXTS: dict = {}
if FULLTEXT_CACHE_PATH.exists():
    with open(FULLTEXT_CACHE_PATH, encoding="utf-8") as _f:
        _FULLTEXTS = json.load(_f)

_PIS: list = []
if PIS_CACHE_PATH.exists():
    with open(PIS_CACHE_PATH, encoding="utf-8") as _f:
        _PIS = json.load(_f)


def _fold(s: str) -> str:
    """Lowercase + strip accents (NFKD). 'Müller' -> 'muller', 'Cortés' -> 'cortes'."""
    if not s:
        return ""
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def _pi_words(pi: dict) -> set[str]:
    """Set of accent-folded word tokens from all searchable fields of a PI.

    Word-token (not substring) matching: 'Müller-Buschbaum' -> {'muller','buschbaum'},
    'Koblmüller' stays {'koblmuller'} — so a 'muller' query no longer pulls Koblmüller.
    """
    parts = [
        pi.get("name", ""),
        pi.get("group", ""),
        pi.get("department", ""),
        pi.get("institution", ""),
        " ".join(pi.get("research_focus", [])),
        " ".join(pi.get("application_fields", [])),
    ]
    folded = _fold(" ".join(parts))
    return set(re.findall(r"\w+", folded))


# Words too common across PI text fields to be useful as match signal.
# Kept tiny on purpose — anything domain-specific (e.g. 'group', 'energy') stays in
# so legitimate queries like 'energy conversion' still work.
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "are", "was", "were",
    "but", "not", "you", "all", "any", "can", "has", "have", "had", "their",
    "they", "them", "its", "into", "over", "such", "than", "then", "these",
    "those", "which", "who", "whom", "what", "when", "where", "why", "how",
})


def _query_tokens(query: str, min_len: int = 3) -> list[str]:
    """Accent-fold the query, split on word boundaries, drop short tokens and stopwords.

    The min-length filter handles trivial noise ('a', 'of', 'in'); the stopword set
    handles common 3+ letter English fillers ('the', 'and', 'for') that otherwise
    match nearly every PI's research description.
    """
    folded = _fold(query)
    return [
        t for t in re.findall(r"\w+", folded)
        if len(t) >= min_len and t not in _STOPWORDS
    ]


def _score_pi(pi: dict, tokens: list[str]) -> int:
    words = _pi_words(pi)
    return sum(1 for t in tokens if t in words)


def _pi_summary(pi: dict) -> dict:
    return {
        "name": pi.get("name"),
        "group": pi.get("group"),
        "institution": pi.get("institution"),
        "research_focus": pi.get("research_focus", []),
        "application_fields": pi.get("application_fields", []),
        "publication_count": len(pi.get("publication_dois", [])),
        "profile_url": pi.get("profile_url"),
    }


@mcp.tool()
def get_paper_by_doi(doi: str) -> str:
    """Return metadata and abstract for a single paper by its DOI.
    Includes full author list, journal, and citation count when present in the OpenAlex metadata cache."""
    doi_key = doi.strip().lower()
    paper = papers_by_doi.get(doi_key)
    if paper is None:
        return json.dumps({"error": f"No paper found for DOI: {doi}"})
    result = dict(paper)
    apply_cache(result, doi_key)
    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
def get_paper_fulltext(doi: str) -> str:
    """Return full text for a single paper by DOI, if available.
    Full text is only available for open-access papers (~65% of the corpus)."""
    doi = doi.strip().lower()
    entry = _FULLTEXTS.get(doi)
    if entry is None:
        return json.dumps({"error": f"No full text cached for DOI: {doi}", "doi": doi})
    return json.dumps({
        "doi": doi,
        "source": entry["source"],
        "url": entry["url"],
        "char_count": entry["char_count"],
        "fetched_at": entry["fetched_at"],
        "fulltext": entry["fulltext"],
    }, indent=2, ensure_ascii=False)


@mcp.tool()
def search_papers(query: str) -> str:
    """Lexical (BM25) search over e-conversion cluster publications.
    Best for exact terminology, acronyms, author names, or any query where the
    user's words are likely to appear verbatim in the title or abstract.
    Returns the top 5 matching papers with titles, authors, abstracts, and linked datasets.
    For conceptual / synonym queries, prefer semantic_search_papers."""
    results = search(query, papers, index)
    return json.dumps(results, indent=2, ensure_ascii=False)


@mcp.tool()
def semantic_search_papers(query: str) -> str:
    """Semantic (embedding) search over e-conversion cluster publications.
    Best for conceptual queries that may use different vocabulary than the
    abstracts — e.g. 'splitting water with sunlight' finding photocatalytic OER
    papers that never use those exact words. For exact terms (acronyms, formulas,
    author names) prefer search_papers; lexical match is usually stronger there.
    Returns the top 5 matching papers ranked by cosine similarity."""
    if not query or not query.strip():
        return json.dumps({"error": "Query must not be empty."})
    if not semantic_search.is_available():
        return json.dumps({
            "error": "Embeddings cache not built. Run: python src/build_embeddings_cache.py",
        })
    results = semantic_search.semantic_search(query, papers_by_doi)
    return json.dumps(results, indent=2, ensure_ascii=False)


@mcp.tool()
def search_pis(query: str) -> str:
    """Search e-conversion PIs (principal investigators) by name, research area, or keyword.
    Returns the top 5 matching PIs with their group, institution, research focus, and publication count.
    Accent-insensitive: 'muller' matches 'Müller-Buschbaum'."""
    if not query or not query.strip():
        return json.dumps({"error": "Query must not be empty."})
    tokens = _query_tokens(query)
    if not tokens:
        return json.dumps({
            "results": [],
            "message": "Query contained no usable tokens (all were too short or stopwords).",
        })
    scored = [(pi, _score_pi(pi, tokens)) for pi in _PIS]
    scored = [(pi, s) for pi, s in scored if s > 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    results = [_pi_summary(pi) for pi, _ in scored[:5]]
    if not results:
        return json.dumps({"results": [], "message": "No PIs matched the query."})
    return json.dumps({"results": results}, indent=2, ensure_ascii=False)


@mcp.tool()
def get_pi(name: str) -> str:
    """Return full profile for a PI by name (last name or full name).
    Includes group, institution, research focus, application fields, website, and their publications
    (titles + abstracts) from the cache. Accent-insensitive: 'Cortes' matches 'Cortés'."""
    if not name or not name.strip():
        return json.dumps({"error": "Name must not be empty."})

    query = _fold(name.strip())

    # 1. Exact last-name match (accent-folded). Collect ALL — two PIs share 'Stein'.
    exact = [pi for pi in _PIS if _fold(pi.get("last_name", "")) == query]
    if len(exact) > 1:
        return json.dumps({
            "error": f"Multiple PIs share the last name '{name}'. Specify the full name.",
            "matches": [_pi_summary(pi) for pi in exact],
        }, indent=2, ensure_ascii=False)
    best = exact[0] if exact else None

    # 2. Substring match on the full (folded) name.
    if best is None:
        subs = [pi for pi in _PIS if query in _fold(pi.get("name", ""))]
        if len(subs) > 1:
            return json.dumps({
                "error": f"Multiple PIs match '{name}'. Specify the full name.",
                "matches": [_pi_summary(pi) for pi in subs],
            }, indent=2, ensure_ascii=False)
        if subs:
            best = subs[0]

    # 3. Keyword scoring across all fields (token-boundary, accent-folded).
    if best is None:
        tokens = _query_tokens(name)
        best_score = 0
        for pi in _PIS:
            s = _score_pi(pi, tokens)
            if s > best_score:
                best_score = s
                best = pi

    if best is None:
        return json.dumps({"error": f"No PI found matching: {name}"})

    result = {
        "name": best.get("name"),
        "group": best.get("group"),
        "department": best.get("department"),
        "institution": best.get("institution"),
        "website": best.get("website"),
        "profile_url": best.get("profile_url"),
        "research_focus": best.get("research_focus", []),
        "application_fields": best.get("application_fields", []),
    }

    dois = best.get("publication_dois", [])
    result["publication_count"] = len(dois)

    # Attach up to 10 papers from the abstract cache
    cached_papers = []
    for doi in dois:
        doi_key = doi.lower()
        paper = papers_by_doi.get(doi_key)
        abstract_entry = _ABSTRACTS.get(doi_key)
        if paper or abstract_entry:
            entry = {}
            if paper:
                entry = {
                    "doi": doi,
                    "title": paper.get("title"),
                    "year": paper.get("year"),
                    "authors": paper.get("authors"),
                }
            apply_cache(entry, doi_key)
            if entry.get("abstract"):
                entry["abstract"] = entry["abstract"][:500]
            cached_papers.append(entry)
        if len(cached_papers) >= 10:
            break

    result["publications_in_cache"] = len(cached_papers)
    result["publications"] = cached_papers
    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
def get_collaborators(pi_query: str) -> str:
    """Return all PIs who share at least one publication with the queried PI, most-shared first.
    Accepts last name, full name, or smid. Use this for 'who does X work with?' questions."""
    if not _graph.is_available():
        return json.dumps({"error": "Collaboration graph not built. Run: python src/scripts/build_graph_cache.py"})
    return json.dumps(_graph.get_collaborators(pi_query), indent=2, ensure_ascii=False)


@mcp.tool()
def joint_papers(pi_a: str, pi_b: str) -> str:
    """Return the DOIs of papers co-authored by two named PIs.
    Returns an empty list if the two PIs have never co-published.
    Accepts last name or full name for each PI."""
    if not _graph.is_available():
        return json.dumps({"error": "Collaboration graph not built. Run: python src/scripts/build_graph_cache.py"})
    return json.dumps(_graph.joint_papers(pi_a, pi_b), indent=2, ensure_ascii=False)


@mcp.tool()
def collaboration_centrality(top_k: int = 10) -> str:
    """Return PIs ranked by betweenness centrality in the co-authorship graph.
    High centrality = a PI bridges otherwise-separate research groups.
    Use for 'who are the connectors / bridges in the cluster?' questions."""
    if not _graph.is_available():
        return json.dumps({"error": "Collaboration graph not built. Run: python src/scripts/build_graph_cache.py"})
    return json.dumps(_graph.collaboration_centrality(top_k), indent=2, ensure_ascii=False)


@mcp.tool()
def collaboration_communities() -> str:
    """Return communities (clusters) of PIs who collaborate internally more than externally.
    Uses greedy modularity detection on the co-authorship graph.
    Use for 'which groups work together?' or cluster-structure questions."""
    if not _graph.is_available():
        return json.dumps({"error": "Collaboration graph not built. Run: python src/scripts/build_graph_cache.py"})
    return json.dumps(_graph.collaboration_communities(), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
