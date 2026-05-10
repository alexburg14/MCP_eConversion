import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from search import load_papers, build_index, search

CSV_PATH = Path(__file__).parent / "data_publication_dois.csv"

mcp = FastMCP("eConversion Papers")

papers = load_papers(CSV_PATH)
bm25 = build_index(papers)


@mcp.tool()
def search_papers(query: str) -> str:
    """Search e-conversion cluster publications by keyword.
    Returns the top 5 matching papers with titles, authors, abstracts, and linked datasets."""
    results = search(query, papers, bm25)
    return json.dumps(results, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
