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

### 7. Built a multi-source full-text pipeline (`src/scripts/build_fulltext_cache.py`)
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

### 8. Scraped the PI / group metadata layer (`src/scripts/build_pis_cache.py`)
Driven by the 2026-05-18 meeting's pivot from "search papers" to "find people and groups". The eConversion website lists 42 PIs at `/members/`, with one `single-staff-page?smid=N` per PI. Both pages are server-side rendered HTML, no API and no JS gating — straight `requests` + regex.

Per PI we capture: name (title/first/last), group, department, institution (TUM/LMU/FHI/MPI FKF), group website, profile image, **Academic Research Focus** bullets, **Fields of Application** tags, and **publication DOIs** harvested from the staff page's Publications section. The DOI list gives PI → papers linkage for free — 34/42 PIs list publications (2,296 unique DOIs across the cluster), which intersects directly with the abstract/full-text caches.

Data gaps surfaced (not parser bugs): three FHI/MPI PIs (Reuter, Roldán Cuenya, Scheurer) have no group field on their listing card — the site collapses group into department for non-TUM/LMU affiliates. Scheurer additionally has no Academic Research Focus and no institution on his profile — captured as empty rather than fabricated.

Output: `data/pis_cache.json`. Wired into the MCP server via `search_pis(query)` and `get_pi(name)` tools (2026-06-05).

### 9. Backfilled OpenAlex metadata (merged into `build_abstracts_cache.py`)
The CSV's `article_authors` column is truncated for non-ASCII surnames (upstream scraper bug), which blocked co-authorship and PI-attribution queries — the 2026-06-05 test report flagged it as the dominant data-side gap. The fix extends `build_abstracts_cache.py`: OpenAlex is now queried once per DOI regardless of whether the `.enl` already had the abstract, and the same cache entry gains `authors` (clean UTF-8), `journal` (from `primary_location.source.display_name`), and `citation_count`. `search.py`'s `apply_cache` overlays abstract + metadata onto every paper returned by `search_papers`, `semantic_search_papers`, `get_paper_by_doi`, and `get_pi`'s linked publications — silent fall-back to the CSV authors when OpenAlex didn't know the DOI.

A separate `metadata_cache.json` was considered and rejected: abstracts and metadata are per-paper facts about the same entity from the same API call, so a second cache/script would have doubled the OpenAlex calls (1007 vs 956) and added enrichment glue for no isolation benefit. One rebuild refreshes abstracts *and* citation counts together. This unblocks "papers in Nature", co-authorship analysis, and citation-aware ranking.

### 10. Hybrid semantic search (`src/scripts/build_embeddings_cache.py`, `src/semantic_search.py`)
The 2026-06-05 report noted BM25 misses conceptual queries that use different vocabulary than the abstracts (e.g. "splitting water with sunlight" → photocatalytic OER). Rather than replacing BM25 — which works well for cluster-internal acronyms, author names, and exact terms — semantic search is added as a *parallel tool* the chat-loop agent can choose per query.

`build_embeddings_cache.py` encodes `"{title}. {abstract}"` for every paper in the abstract cache with `BAAI/bge-small-en-v1.5` (384-d, L2-normalised) into `data/embeddings_cache.npz`. ~950 docs fit in memory; no vector DB. `semantic_search.py` lazy-loads the npz + model on first call so MCP startup stays cheap, and reuses the same metadata enrichment as the BM25 path. The new MCP tool is `semantic_search_papers(query)`. The chat-app system prompt now describes when to pick lexical vs semantic, and explicitly tells the agent to run *both* when unsure to widen recall before synthesizing.

This sits inside the [[project-direction]] "vectorless RAG by default" stance: PI lookups and exact-term searches stay lexical/structured; embeddings are one option among several rather than the default retrieval layer.

---

## What Worked
- MCP server is live and functional — tested successfully with real queries
- Abstract cache gives near-complete coverage instantly, without API dependency
- The `.enl` turned out to be a much richer local source than expected
- Two-stage BM25 cascade handles both precise and conceptual queries better than title-only search
- `get_paper_by_doi` enables direct lookup without needing to search

## What Didn't Work / Limitations
- **4 papers** have no abstract available from any source
- ~~**Author names are truncated**~~ Resolved 2026-06-12: OpenAlex `authorships` are overlaid onto all results (see step 9). The CSV column itself is still truncated (upstream scraper bug) but no longer surfaces anywhere. Note: OpenAlex names contain Unicode-hyphen variants (U+2010 vs ASCII `-`) — normalize before exact author matching / co-authorship graphs.
- ~~**BM25 is still lexical**~~ Resolved 2026-06-12: `semantic_search_papers` (BGE-small embeddings) covers conceptual/synonym queries as a parallel tool (see step 10).
- **Single-user setup** — currently runs locally on one machine, not accessible cluster-wide
- **Full-text coverage is partial** — 402/956 (42%) cached. The 554 remaining are dominated by DFG/ERC-funded EU chemistry/physics with no preprint and no PMC deposit; would need institutional access to recover. Wiley/ACS-without-NIH-funding is the dominant failure mode.
- **Full text is not yet wired into search** — `fulltext_cache.json` is populated but `search_papers` still searches only abstracts. Next step: either extend BM25 to full text or let the LLM call `get_paper_fulltext` for top candidates.

---

## What Comes Next

### Short term
- ~~**Wire PIs into the MCP server**~~ Done 2026-06-05: `search_pis(query)` and `get_pi(name)` are live. The "Wer ist Patrick Rinke?" user story works end-to-end.
- ~~**Fix author names**~~ Done 2026-06-12: `build_abstracts_cache.py` now also pulls full authors / journal / citation count from OpenAlex into each cache entry, overlaid onto all paper results.
- ~~**Hybrid semantic search**~~ Done 2026-06-12: `semantic_search_papers` available alongside `search_papers`; agent picks per query.
- **Add more MCP tools**: `list_papers(year, author)` for browsing
- **Keep caches fresh**: run `build_abstracts_cache.py` and `build_pis_cache.py` when the website changes (the former also refreshes citation counts); rebuild `embeddings_cache.npz` after abstract refresh

### Medium term
- **Wire full text into search**: currently `search_papers` only searches abstracts. Either extend BM25 to also index `fulltext_cache.json` (now 402 papers, avg ~40k chars each), or have the LLM call `get_paper_fulltext(doi)` for top-ranked abstract hits. With 138 of the new entries being PMC JATS-derived markdown (cleanly section-segmented), a `get_paper_section(doi, "methods")` tool becomes feasible.
- **Migrate from JSON to a database**: SQLite is the natural first step (same format as the `.enl`). PostgreSQL when cluster-wide concurrent access is needed.
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
| `src/server.py` | MCP server — exposes `search_papers`, `semantic_search_papers`, `get_paper_by_doi`, `get_paper_fulltext`, `search_pis`, `get_pi` |
| `src/search.py` | Two-stage BM25 search (title → abstract fallback) + abstracts/metadata cache reader |
| `src/semantic_search.py` | Cosine search over BGE-small embeddings; lazy-loads model on first call |
| `src/scripts/build_abstracts_cache.py` | Rebuilds abstracts cache (.enl → OpenAlex → S2) + OpenAlex authors / journal / citation_count per entry |
| `src/scripts/build_embeddings_cache.py` | Encodes title+abstract with BGE-small into `data/embeddings_cache.npz` |
| `src/scripts/build_fulltext_cache.py` | Multi-source full-text pipeline: arXiv → PMC (JATS XML) → repos → publisher PDFs → HTML fallback |
| `src/scripts/build_pis_cache.py` | Scrapes `e-conversion.de/members/` and per-PI staff pages into `data/pis_cache.json` |
| `data/abstracts_cache.json` | One entry per DOI: abstract + OpenAlex authors / journal / citation_count |
| `data/embeddings_cache.npz` | 956 × 384 BGE-small vectors + parallel DOI array |
| `data/fulltext_cache.json` | 402 full-text bodies keyed by DOI (source: pdf / html / pmc) |
| `data/pis_cache.json` | 42 PIs keyed by smid (name, group, dept, institution, research focus, application fields, publication DOIs) |
| `data/data_publication_dois.csv` | 956 papers + 148 dataset links |
| `data/e-conversion-Converted.enl` | Source EndNote library (SQLite) |
| `data/scraper_cache.json` | Raw scraper output from e-conversion.de |