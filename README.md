# eConversion Papers MCP Server

A local MCP (Model Context Protocol) server that exposes the e-conversion research cluster's publication list as searchable tools for Claude Code. Built to make knowledge transfer inside the cluster easier — researchers can ask natural-language questions and find relevant work from within the cluster.

## What it does

- **956 publications** scraped from `e-conversion.de/publikationen`
- **952 abstracts (99.6% coverage)** cached locally — no API calls at search time
- **148 dataset links** (e.g. crystal structures in CSD/CCDC) attached to their parent papers
- **Two-stage BM25 search** — searches titles first, falls back to abstracts when the abstract index scores higher (handles both precise and conceptual queries)

## MCP tools

| Tool | Description |
|---|---|
| `search_papers(query)` | Returns the top 5 papers matching the query, with titles, authors, year, abstracts, and any linked datasets. Each result includes a `matched_on` field (`title` or `abstract`) indicating which index fired. |
| `get_paper_by_doi(doi)` | Direct lookup — returns full metadata and abstract for a single paper. |

## Setup

```bash
pip install -r requirements.txt
```

The server is registered via `.mcp.json` and loads automatically in Claude Code from this directory.

To run it manually:

```bash
python server.py
```

## Rebuilding the abstract cache

When new papers are added to the publication list, regenerate `abstracts_cache.json`:

```bash
python build_abstracts_cache.py
```

This pulls abstracts from the local `.enl` EndNote library first (905 papers), then falls back to the OpenAlex API for any remaining DOIs.

## Files

| File | Purpose |
|---|---|
| `server.py` | MCP server entry point — exposes the two tools |
| `search.py` | Two-stage BM25 search engine + abstract cache reader |
| `build_abstracts_cache.py` | Rebuilds `abstracts_cache.json` from `.enl` + OpenAlex fallback |
| `abstracts_cache.json` | 952 abstracts keyed by DOI |
| `data_publication_dois.csv` | 956 papers + 148 dataset links |
| `e-conversion-Converted.enl` | Source EndNote library (SQLite format) |
| `scraper_cache.json` | Raw scraper output from e-conversion.de |
| `.mcp.json` | Claude Code MCP server registration |
