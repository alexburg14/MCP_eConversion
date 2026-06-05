import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from search import load_papers, build_index, search, _ABSTRACTS

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


def _pi_text(pi: dict) -> str:
    """Concatenate all searchable fields of a PI into one lowercase string."""
    parts = [
        pi.get("name", ""),
        pi.get("group", ""),
        pi.get("department", ""),
        pi.get("institution", ""),
        " ".join(pi.get("research_focus", [])),
        " ".join(pi.get("application_fields", [])),
    ]
    return " ".join(parts).lower()


def _score_pi(pi: dict, tokens: list[str]) -> int:
    text = _pi_text(pi)
    return sum(1 for t in tokens if t in text)


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
    """Return metadata and abstract for a single paper by its DOI."""
    paper = papers_by_doi.get(doi.strip().lower())
    if paper is None:
        return json.dumps({"error": f"No paper found for DOI: {doi}"})
    result = dict(paper)
    cached = _ABSTRACTS.get(result["doi"].lower())
    if cached:
        result["abstract"] = cached["abstract"]
        result["abstract_source"] = cached["source"]
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
    """Search e-conversion cluster publications by keyword.
    Returns the top 5 matching papers with titles, authors, abstracts, and linked datasets."""
    results = search(query, papers, index)
    return json.dumps(results, indent=2, ensure_ascii=False)


@mcp.tool()
def search_pis(query: str) -> str:
    """Search e-conversion PIs (principal investigators) by name, research area, or keyword.
    Returns the top 5 matching PIs with their group, institution, research focus, and publication count."""
    tokens = query.lower().split()
    if not tokens:
        return json.dumps({"error": "Query must not be empty."})
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
    (titles + abstracts) from the cache."""
    query = name.strip().lower()
    best = None
    best_score = 0

    # Prefer exact last-name match
    for pi in _PIS:
        if pi.get("last_name", "").lower() == query:
            best = pi
            break

    # Fall back to substring match on full name
    if best is None:
        for pi in _PIS:
            if query in pi.get("name", "").lower():
                best = pi
                break

    # Fall back to keyword scoring across all fields
    if best is None:
        tokens = query.split()
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
        paper = papers_by_doi.get(doi.lower())
        abstract_entry = _ABSTRACTS.get(doi.lower())
        if paper or abstract_entry:
            entry = {}
            if paper:
                entry = {
                    "doi": doi,
                    "title": paper.get("title"),
                    "year": paper.get("year"),
                    "authors": paper.get("authors"),
                }
            if abstract_entry:
                entry["abstract"] = abstract_entry.get("abstract", "")[:500]
            cached_papers.append(entry)
        if len(cached_papers) >= 10:
            break

    result["publications_in_cache"] = len(cached_papers)
    result["publications"] = cached_papers
    return json.dumps(result, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
