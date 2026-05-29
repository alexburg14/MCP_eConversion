# eConversion Knowledge Transfer — Project Context

## Objective
Make knowledge transfer inside the e-conversion research cluster easier by turning the cluster's publication list into a FAIR, AI-queryable knowledge base. Researchers should be able to ask natural language questions and find relevant work from within the cluster.

---

## What We Did

### 1. Understood the data landscape
- **`scraper_cache.json` / `data_publication_dois.csv`**: 956 unique cluster publications scraped from `e-conversion.de/publikationen` via the [`datapublicationlist`](https://github.com/harrytyp/datapublicationlist) tool. The CSV also links papers to their deposited datasets (148 rows have dataset DOIs, e.g. crystal structures in CSD/CCDC).
- **`e-conversion-Converted.enl`**: An EndNote library in SQLite format. 1,584 records — a broader reference library that overlaps with but is not identical to the website list.

### 2. Built an MCP server
A Claude Code MCP server (`server.py` + `search.py`) exposing three tools, registered via `.mcp.json` so they load automatically in Claude Code:
- `search_papers(query)` — keyword search returning top 5 results with abstracts
- `get_paper_by_doi(doi)` — direct lookup returning full metadata and abstract for a single paper
- `get_paper_fulltext(doi)` — returns cached full-text body for OA papers (see step 7)

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

### 7. Built a multi-source full-text pipeline (`src/build_fulltext_cache.py`)
Per-DOI pipeline that queries four OA sources plus NCBI's PMC API, deduplicates URLs, and tries paths in tier order:

| Tier | Source | Why |
|---|---|---|
| 0 | **arXiv** (`arxiv.org/pdf/{id}`) | No bot-blocking, clean PDFs. arXiv ID is harvested from OpenAlex `ids.arxiv`, Semantic Scholar `externalIds.ArXiv`, and (last resort) arXiv's own `query?search_query=doi:` API — S2 returns it even when `openAccessPdf` is empty, which is the key insight |
| 0.5 | **PMC** (NCBI eutils efetch on `db=pmc`) | Structured JATS XML, no bot-blocking. The big rescue: papers funded by NIH/HHMI must be deposited in PubMed Central via the NIH Public Access Policy, so PMC holds free copies of many ACS/Wiley/JACS/Nature articles that the publisher still paywalls. PMCID is discovered via OpenAlex `ids.pmcid` or NCBI's idconv endpoint. JATS XML is rendered to markdown with a small ElementTree walker (sections + abstract + body; skips figures/tables/formulas). 138 of our 956 papers were recovered this way alone |
| 1 | **Repository PDFs** (institutional repos, hal., edoc., pure., osti, etc.) | Backup copies that survive when publisher PDFs 403 |
| 2 | **Publisher PDFs** (Nature, RSC, AIP usually OK; Wiley/ACS often 403) | Tried last because of bot-blocking |
| (fallback) | HTML via `trafilatura` | Some publishers inline the full body in HTML (Nature OA, some ACS OA, RSC, IOP). `MIN_CHARS=3000` filters out abstract-only landing pages |

**Tools chosen:**
- PDF → markdown: `pymupdf4llm` (lightweight, no ML models, ~50k chars per typical paper)
- HTML → markdown: `trafilatura` (extracts main article body, strips boilerplate)

**Quality safeguards:**
- Browser User-Agent (PMC/Wiley/ACS reject our default UA)
- `%PDF` magic-byte verification (PMC returns HTML interstitials disguised as `.pdf` URLs)
- Bare-URL retry (Unpaywall returns `science.org/...?download=true` which returns 403; same URL without query string returns the PDF)
- `MIN_CHARS=3000` floor — full articles are 10k+ chars; this rejects landing pages with only the abstract
- Deterministic DOI shuffle (random seed 42) — avoids hammering one publisher in alphabetical order
- Incremental cache writes every 50 papers (resumable after interruption)
- Source attribution stored per entry (`source_origin`: arxiv | repository | publisher) for later auditing

**Smoke test (30-DOI random sample, seed 42)**: 60% success rate vs. 40% with single-source Unpaywall-only. arXiv recovered papers like `10.1103/physrevb.110.125202` that Unpaywall and OpenAlex both marked "closed".

**Full run status**: **402/956 (42.1%)** papers cached. Two passes: an initial run reached 244/956 (mostly arXiv + repository PDFs + publisher HTML), then a second pass after adding PMC tier 0.5 contributed +138 net new hits (almost all PMC), plus 1 HTML. Breakdown of the full cache: pdf 186 (arxiv 49, repository 27, publisher ~110), html 78, pmc 138. The 554 still-missing DOIs are dominated by DFG/ERC-funded EU chemistry/physics with no preprint and no PMC deposit — would need institutional access to recover. The arXiv-by-DOI fallback was added but contributed 0 hits on the residue (papers not exposed by OpenAlex/S2's arXiv metadata aren't found by arXiv's own DOI search either — kept anyway, costs one extra request when arXiv-empty).

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
- **Full-text coverage is partial** — 402/956 (42%) cached. The 554 remaining are dominated by DFG/ERC-funded EU chemistry/physics with no preprint and no PMC deposit; would need institutional access to recover. Wiley/ACS-without-NIH-funding is the dominant failure mode.
- **Full text is not yet wired into search** — `fulltext_cache.json` is populated but `search_papers` still searches only abstracts. Next step: either extend BM25 to full text or let the LLM call `get_paper_fulltext` for top candidates.

---

## What Comes Next

### Short term
- **Add more MCP tools**: `list_papers(year, author)` for browsing
- **Fix author names**: fetch full author lists from OpenAlex to replace truncated names in the CSV
- **Keep cache fresh**: run `build_abstracts_cache.py` when new papers are published

### Medium term
- **Wire full text into search**: currently `search_papers` only searches abstracts. Either extend BM25 to also index `fulltext_cache.json` (now 402 papers, avg ~40k chars each), or have the LLM call `get_paper_fulltext(doi)` for top-ranked abstract hits. With 138 of the new entries being PMC JATS-derived markdown (cleanly section-segmented), a `get_paper_section(doi, "methods")` tool becomes feasible.
- **Migrate from JSON to a database**: SQLite is the natural first step (same format as the `.enl`). PostgreSQL when cluster-wide concurrent access is needed.
- **Semantic search**: add vector embeddings (e.g. `allenai/scibert`) as a third stage after the two-stage BM25 cascade, to handle synonym/conceptual queries. ~956 abstracts is small enough to embed cheaply.
- **PDF extractor upgrade**: PyMuPDF4LLM is sufficient for plain prose. For papers with complex layouts/equations/tables, upgrade to Marker (ML-based, surya models ~1-2GB) or MinerU (heaviest, best quality).

### Long term
- **Cluster-wide deployment**: shared server or cloud-hosted endpoint so all researchers can query without running anything locally
- **Web interface**: a simple search UI on top of the database for non-technical users
- **Automated updates**: hook into the e-conversion website to detect new publications and update the cache automatically

---

## Key Files
Repo is split into `src/` (code, tracked) and `data/` (caches and source data, gitignored).

| File | Purpose |
|---|---|
| `src/server.py` | MCP server — exposes `search_papers`, `get_paper_by_doi`, `get_paper_fulltext` |
| `src/search.py` | Two-stage BM25 search (title → abstract fallback) + abstract cache reader |
| `src/build_abstracts_cache.py` | Rebuilds abstracts cache from .enl + API fallback |
| `src/build_fulltext_cache.py` | Multi-source full-text pipeline: arXiv → PMC (JATS XML) → repos → publisher PDFs → HTML fallback |
| `data/abstracts_cache.json` | 952 abstracts keyed by DOI |
| `data/fulltext_cache.json` | 402 full-text bodies keyed by DOI (source: pdf / html / pmc) |
| `data/data_publication_dois.csv` | 956 papers + 148 dataset links |
| `data/e-conversion-Converted.enl` | Source EndNote library (SQLite) |
| `data/scraper_cache.json` | Raw scraper output from e-conversion.de |