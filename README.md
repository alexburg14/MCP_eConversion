# eConversion Papers MCP Server

A local MCP (Model Context Protocol) server that exposes the e-conversion research cluster's publication list as searchable tools for Claude Code. Built to make knowledge transfer inside the cluster easier — researchers can ask natural-language questions and find relevant work from within the cluster.

## What it does

- **956 publications** scraped from `e-conversion.de/publikationen`
- **953 abstracts (99.7% coverage)** cached locally — no API calls at search time
- **402 full-text bodies (42.1% coverage)** cached locally, harvested from arXiv, PMC (NIH-deposited author manuscripts in JATS XML), institutional repositories, publisher PDFs, and HTML landing pages
- **148 dataset links** (e.g. crystal structures in CSD/CCDC) attached to their parent papers
- **42 PIs** scraped from `e-conversion.de/members/` with group, department, institution, research focus, application fields, and (where listed) their publication DOIs — links each PI to their papers in the cache
- **Two-stage BM25 search** — searches titles first, falls back to abstracts when the abstract index scores higher (handles both precise and conceptual queries)
- **Semantic search** — BGE-small embeddings over titles + abstracts as a parallel retrieval path for conceptual / synonym queries that lexical BM25 misses
- **OpenAlex metadata** — full author lists (replaces truncated CSV names), journal/venue, and citation counts, stored alongside each abstract

## MCP tools

| Tool | Description |
|---|---|
| `search_papers(query)` | Lexical (BM25) search — top 5 papers with titles, authors, year, abstracts, and any linked datasets. Best for exact terminology, acronyms, author names. `matched_on` (`title` / `abstract`) indicates which index fired. |
| `semantic_search_papers(query)` | Semantic (embedding) search — top 5 papers ranked by cosine similarity over BGE-small embeddings. Best for conceptual queries where wording may differ from the abstracts. |
| `get_paper_by_doi(doi)` | Direct lookup — returns full metadata and abstract for a single paper. When OpenAlex metadata is cached, includes the full author list, journal, and citation count. |
| `get_paper_fulltext(doi)` | Returns cached full-text markdown for a single paper, with `source` (`pdf` / `html` / `pmc`), origin URL, char count, and fetch date. Only available for the ~42% of papers covered by the full-text cache. |
| `search_nomad(elements, formula, author, text)` | Live search over the **public NOMAD** materials repository — external computed/experimental data, not cluster papers. Filters combine with AND. Text queries are relevance-ranked (`_score`), structured filters newest-first. Returns `total_matches` plus a sample. |
| `search_pis(query)` | Keyword search across PI names, groups, research focus, and application fields. Returns the top 5 matching PIs with group, institution, research focus, and publication count. |
| `get_pi(name)` | Profile lookup for a PI by last name, full name, or keyword. Returns full details plus up to 10 linked papers from the abstract cache. |
| `list_papers(author, year, journal, limit)` | Exhaustive metadata filtering (not top-5 ranking) — every paper matching the given filters, newest first. At least one of `author` / `year` / `journal` is required; filters combine with AND. |
| `get_collaborators(pi_query)` | All PIs who share at least one publication with the queried PI, most-shared first. |
| `joint_papers(pi_a, pi_b)` | DOIs of papers co-authored by two named PIs. |
| `collaboration_centrality(top_k)` | PIs ranked by betweenness centrality in the co-authorship graph — the connectors between otherwise-separate groups. |
| `collaboration_communities()` | Clusters of PIs who collaborate internally more than externally (greedy modularity detection). |

## Setup

```bash
pip install -r requirements.txt
```

The server is registered via `.mcp.json` and loads automatically in Claude Code from this directory.

To run it manually:

```bash
python src/server.py
```

## Configuration

Cluster-specific settings — the cluster's name and description, the LLM endpoint,
and the selectable models — live in [`config.toml`](config.toml) at the repo root.
Adapting the assistant to a different research cluster is a matter of editing that
one file; no Python changes are required. Publication and PI counts shown in the UI
and system prompt are derived from the loaded caches, so they stay correct
automatically.

## Chat interface

A Streamlit web app that lets researchers ask questions in natural language. It calls the search tools internally and uses an LLM from the GWDG SAIA / Academic Cloud Chat AI endpoint (OpenAI-compatible, `https://chat-ai.academiccloud.de/v1`) to synthesize answers. The model is selectable in the sidebar (default `qwen3.5-122b-a10b`; all listed models verified for tool calling).

Put the SAIA key in `.env` at the repo root (`API_KEY=...`) or export it, then:

```bash
streamlit run src/app.py
```

SAIA keys expire after 6 months — renew at https://saia.gwdg.de/dashboard.

## Layout

Runtime code (server, search, chat app) lives under `src/`; one-shot build and extract scripts live under `src/scripts/`; data caches and source files live under `data/`. The `data/` directory is gitignored — caches must be built locally (see below).

## Rebuilding the caches

**Abstracts + metadata** — when new papers are added, or to refresh citation counts:

```bash
python src/scripts/build_abstracts_cache.py
```

Abstracts come from the local `.enl` EndNote library first (905 papers), then OpenAlex, then Semantic Scholar. OpenAlex is queried once per DOI regardless, supplying the full author list (the CSV's author column is truncated for non-ASCII surnames), journal, and citation count in the same cache entry. Full rebuild, ~956 API calls, 3–5 min.

**Full text** — incremental, resumable:

```bash
python src/scripts/build_fulltext_cache.py
```

Per-DOI pipeline that tries sources in tier order: arXiv → PMC (NCBI eutils, JATS XML) → repository PDFs → publisher PDFs → HTML fallback (`trafilatura`). Already-cached DOIs are skipped; only NOT-FOUND DOIs are retried. Writes a checkpoint every 50 papers so the run is safe to interrupt.

**PIs** — scrapes the eConversion members directory and each individual staff page:

```bash
python src/scripts/build_pis_cache.py
```

Writes `data/pis_cache.json` with one entry per PI. Re-run anytime the members list changes; pass `--force` to refresh existing entries.

**Embeddings** — dense vectors for `semantic_search_papers`:

```bash
python src/scripts/build_embeddings_cache.py
```

Encodes `title + abstract` for every paper in the abstract cache with `BAAI/bge-small-en-v1.5` (384-d, L2-normalised) and writes `data/embeddings_cache.npz`. Re-run only when the abstract cache changes. First run downloads the model (~130 MB).

**Collaboration graph** — PI co-authorship graph for the `get_collaborators` / `joint_papers` / `collaboration_centrality` / `collaboration_communities` tools:

```bash
python src/scripts/build_graph_cache.py
```

Two PIs are linked iff they share a publication DOI (factual set intersection over `pis_cache.json`, no name disambiguation needed). Writes `data/collaboration_graph.json`. Re-run only when `pis_cache.json` changes.

**Proposal summary** — one-shot extraction of Section 2 of the e-conversion 2.0 DFG proposal, used as system-prompt context in the chat interface:

```bash
python src/scripts/extract_proposal_summary.py
```

Reads `data/EXC_2089_e-conversion_A_Proposal_R.pdf` and writes `data/proposal_summary.md` (~1.5K tokens). Re-run only if the proposal PDF changes.

## Files

| File | Purpose |
|---|---|
| `config.toml` | Cluster-specific settings (name, description, LLM endpoint, models) — the one file to edit when adapting the template |
| `src/config.py` | Loads `config.toml` once into a frozen, typed `Config` object via `get_config()` |
| `src/server.py` | MCP server entry point — the single source of truth for all tool definitions |
| `src/openai_tools.py` | Derives the chat app's OpenAI tool schemas and dispatch from the MCP registry — no hand-maintained duplicate |
| `src/search.py` | Two-stage BM25 search engine + abstracts/metadata cache reader |
| `src/semantic_search.py` | BGE-small cosine search; lazy-loads model and embeddings on first call |
| `src/nomad_search.py` | Live query against the public NOMAD API (no cache, no credentials) |
| `src/scripts/build_abstracts_cache.py` | Rebuilds `data/abstracts_cache.json`: abstracts (`.enl` → OpenAlex → S2) plus OpenAlex authors / journal / citation count |
| `src/scripts/build_embeddings_cache.py` | Encodes title + abstract with BGE-small into `data/embeddings_cache.npz` |
| `src/scripts/build_fulltext_cache.py` | Multi-source full-text builder (arXiv → PMC → repos → publisher PDFs → HTML) |
| `src/scripts/build_pis_cache.py` | Scrapes `e-conversion.de/members/` and individual staff pages into `data/pis_cache.json` |
| `src/scripts/build_graph_cache.py` | Builds the PI co-authorship graph into `data/collaboration_graph.json` |
| `src/scripts/extract_proposal_summary.py` | Extracts Section 2 of the e-conversion 2.0 proposal PDF into `data/proposal_summary.md` |
| `src/graph.py` | Loads `collaboration_graph.json` into networkx; backs the collaboration-graph tools |
| `data/abstracts_cache.json` | One entry per DOI: abstract + OpenAlex authors / journal / citation_count |
| `data/embeddings_cache.npz` | Parallel `dois` + 384-d `vectors` arrays for semantic search |
| `data/fulltext_cache.json` | 402 full-text bodies keyed by DOI |
| `data/pis_cache.json` | 42 PIs keyed by smid (group, dept, institution, research focus, publication DOIs) |
| `data/collaboration_graph.json` | Node-link JSON of the PI co-authorship graph |
| `data/data_publication_dois.csv` | 956 papers + 148 dataset links |
| `data/e-conversion-Converted.enl` | Source EndNote library (SQLite format) |
| `data/scraper_cache.json` | Raw scraper output from e-conversion.de |
| `data/EXC_2089_e-conversion_A_Proposal_R.pdf` | Source PDF of the e-conversion 2.0 DFG proposal |
| `data/proposal_summary.md` | Section 2 of the proposal, extracted for chat-interface system context |
| `.mcp.json` | Claude Code MCP server registration |
