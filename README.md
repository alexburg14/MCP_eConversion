# eConversion Papers MCP Server

A local MCP (Model Context Protocol) server that exposes the e-conversion research cluster's publication list as searchable tools for Claude Code. Built to make knowledge transfer inside the cluster easier — researchers can ask natural-language questions and find relevant work from within the cluster.

## What it does

- **956 publications** scraped from `e-conversion.de/publikationen`
- **952 abstracts (99.6% coverage)** cached locally — no API calls at search time
- **402 full-text bodies (42.1% coverage)** cached locally, harvested from arXiv, PMC (NIH-deposited author manuscripts in JATS XML), institutional repositories, publisher PDFs, and HTML landing pages
- **148 dataset links** (e.g. crystal structures in CSD/CCDC) attached to their parent papers
- **Two-stage BM25 search** — searches titles first, falls back to abstracts when the abstract index scores higher (handles both precise and conceptual queries)

## MCP tools

| Tool | Description |
|---|---|
| `search_papers(query)` | Returns the top 5 papers matching the query, with titles, authors, year, abstracts, and any linked datasets. Each result includes a `matched_on` field (`title` or `abstract`) indicating which index fired. |
| `get_paper_by_doi(doi)` | Direct lookup — returns full metadata and abstract for a single paper. |
| `get_paper_fulltext(doi)` | Returns cached full-text markdown for a single paper, with `source` (`pdf` / `html` / `pmc`), origin URL, char count, and fetch date. Only available for the ~42% of papers covered by the full-text cache. |

## Setup

```bash
pip install -r requirements.txt
```

The server is registered via `.mcp.json` and loads automatically in Claude Code from this directory.

To run it manually:

```bash
python src/server.py
```

## Layout

Code lives under `src/`; data caches and source files live under `data/`. The `data/` directory is gitignored — caches must be built locally (see below).

## Rebuilding the caches

**Abstracts** — when new papers are added to the publication list:

```bash
python src/build_abstracts_cache.py
```

Pulls from the local `.enl` EndNote library first (905 papers), then OpenAlex for any remaining DOIs.

**Full text** — incremental, resumable:

```bash
python src/build_fulltext_cache.py
```

Per-DOI pipeline that tries sources in tier order: arXiv → PMC (NCBI eutils, JATS XML) → repository PDFs → publisher PDFs → HTML fallback (`trafilatura`). Already-cached DOIs are skipped; only NOT-FOUND DOIs are retried. Writes a checkpoint every 50 papers so the run is safe to interrupt.

## Files

| File | Purpose |
|---|---|
| `src/server.py` | MCP server entry point — exposes the three tools |
| `src/search.py` | Two-stage BM25 search engine + abstract cache reader |
| `src/build_abstracts_cache.py` | Rebuilds `data/abstracts_cache.json` from `.enl` + OpenAlex fallback |
| `src/build_fulltext_cache.py` | Multi-source full-text builder (arXiv → PMC → repos → publisher PDFs → HTML) |
| `data/abstracts_cache.json` | 952 abstracts keyed by DOI |
| `data/fulltext_cache.json` | 402 full-text bodies keyed by DOI |
| `data/data_publication_dois.csv` | 956 papers + 148 dataset links |
| `data/e-conversion-Converted.enl` | Source EndNote library (SQLite format) |
| `data/scraper_cache.json` | Raw scraper output from e-conversion.de |
| `.mcp.json` | Claude Code MCP server registration |
