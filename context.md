# eConversion Knowledge Transfer — Project Context

## Objective
Make knowledge transfer inside the e-conversion research cluster easier by turning the cluster's publication list into a FAIR, AI-queryable knowledge base. Researchers should be able to ask natural language questions and find relevant work from within the cluster.

---

## What We Did

### 1. Understood the data landscape
- **`scraper_cache.json` / `data_publication_dois.csv`**: 956 unique cluster publications scraped from `e-conversion.de/publikationen` via the [`datapublicationlist`](https://github.com/harrytyp/datapublicationlist) tool. The CSV also links papers to their deposited datasets (148 rows have dataset DOIs, e.g. crystal structures in CSD/CCDC).
- **`e-conversion-Converted.enl`**: An EndNote library in SQLite format. 1,584 records — a broader reference library that overlaps with but is not identical to the website list.

### 2. Built an MCP server
A Claude Code MCP server (`server.py` + `search.py`) exposing two tools, registered via `.mcp.json` so they load automatically in Claude Code:
- `search_papers(query)` — keyword search returning top 5 results with abstracts
- `get_paper_by_doi(doi)` — direct lookup returning full metadata and abstract for a single paper

### 3. Cleaned the data
Fixed LaTeX accent escapes in author/title fields (e.g. `Rodr\'{i}` → `Rodrí`) across 559 rows of the CSV. The underlying bug is in the upstream scraper's regex (`[^\}]+` stops at the first `}` of a LaTeX accent group) — partially recoverable but surnames remain truncated.

### 4. Benchmarked abstract sources
Tested 4 APIs on a 50-DOI sample:
| Source | Coverage | Notes |
|---|---|---|
| OpenAlex | 86% | Best single API source |
| Semantic Scholar | 80% | Good fallback |
| Europe PMC | 60% | Life-sciences bias |
| Crossref | 36% | Low because ACS/Wiley/RSC don't deposit abstracts |
| Union of all 4 | 96% | Ceiling without HTML scraping |

### 5. Built a local abstract cache
Discovered the `.enl` already contains 905 of our 956 papers with high-quality abstracts (avg 1,293 chars vs ~1,034 from OpenAlex). Built `abstracts_cache.json` using:
- `.enl` as primary source (905 papers)
- OpenAlex API as fallback for the 51 missing DOIs (47 found)
- **Final coverage: 952/956 papers (99.6%)**, zero API calls at search time

### 6. Audited full-text availability
Checked Unpaywall for all 956 DOIs:
- **628 papers (65.7%)** are Open Access in some form
- **412 papers (43.1%)** have a direct PDF URL available
- **327 papers (34.2%)** are fully closed access — no legal free full-text

---

## What Worked
- MCP server is live and functional — tested successfully with real queries
- Abstract cache gives near-complete coverage instantly, without API dependency
- The `.enl` turned out to be a much richer local source than expected
- Two-stage BM25 cascade handles both precise and conceptual queries better than title-only search
- `get_paper_by_doi` enables direct lookup without needing to search

## What Didn't Work / Limitations
- **4 papers** have no abstract available from any source
- **Author names are truncated** for papers with non-ASCII characters — this is a bug in the upstream scraper that would require a fix + re-scrape to fully resolve
- **BM25 is still lexical** — conceptual queries with different vocabulary than the abstracts (e.g. synonyms) won't match; this requires semantic/embedding search to solve fully
- **Single-user setup** — currently runs locally on one machine, not accessible cluster-wide
- **No full-text content** — only abstracts; the 412 OA PDFs are identified but not yet fetched or indexed

---

## What Comes Next

### Short term
- **Add more MCP tools**: `list_papers(year, author)` for browsing
- **Fix author names**: fetch full author lists from OpenAlex to replace truncated names in the CSV
- **Keep cache fresh**: run `build_abstracts_cache.py` when new papers are published

### Medium term
- **Migrate from JSON to a database**: SQLite is the natural first step (same format as the `.enl`). PostgreSQL when cluster-wide concurrent access is needed
- **Semantic search**: add vector embeddings (e.g. `allenai/scibert`) as a third stage after the two-stage BM25 cascade, to handle synonym/conceptual queries. ~956 abstracts is small enough to embed cheaply
- **Full-text pipeline**: download the 412 OA PDFs via Unpaywall URLs, extract text with `pdfplumber`, add to the index

### Long term
- **Cluster-wide deployment**: shared server or cloud-hosted endpoint so all researchers can query without running anything locally
- **Web interface**: a simple search UI on top of the database for non-technical users
- **Automated updates**: hook into the e-conversion website to detect new publications and update the cache automatically

---

## Key Files
| File | Purpose |
|---|---|
| `server.py` | MCP server — exposes `search_papers` and `get_paper_by_doi` tools |
| `search.py` | Two-stage BM25 search (title → abstract fallback) + abstract cache reader |
| `abstracts_cache.json` | 952 abstracts keyed by DOI |
| `build_abstracts_cache.py` | Rebuilds the cache from .enl + API fallback |
| `data_publication_dois.csv` | 956 papers + 148 dataset links |
| `e-conversion-Converted.enl` | Source EndNote library (SQLite) |
| `scraper_cache.json` | Raw scraper output from e-conversion.de |