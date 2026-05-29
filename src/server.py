import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from search import load_papers, build_index, search, _ABSTRACTS

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = _DATA_DIR / "data_publication_dois.csv"
FULLTEXT_CACHE_PATH = _DATA_DIR / "fulltext_cache.json"

mcp = FastMCP("eConversion Papers")

papers = load_papers(CSV_PATH)
index = build_index(papers)
papers_by_doi = {p["doi"].lower(): p for p in papers}

_FULLTEXTS: dict = {}
if FULLTEXT_CACHE_PATH.exists():
    with open(FULLTEXT_CACHE_PATH, encoding="utf-8") as _f:
        _FULLTEXTS = json.load(_f)


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


if __name__ == "__main__":
    mcp.run()
