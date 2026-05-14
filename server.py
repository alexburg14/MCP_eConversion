import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from search import load_papers, build_index, search, _ABSTRACTS

CSV_PATH = Path(__file__).parent / "data_publication_dois.csv"

mcp = FastMCP("eConversion Papers")

papers = load_papers(CSV_PATH)
index = build_index(papers)
papers_by_doi = {p["doi"].lower(): p for p in papers}


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
def search_papers(query: str) -> str:
    """Search e-conversion cluster publications by keyword.
    Returns the top 5 matching papers with titles, authors, abstracts, and linked datasets."""
    results = search(query, papers, index)
    return json.dumps(results, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
