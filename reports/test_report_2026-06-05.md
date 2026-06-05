# MCP Server Test Report — 2026-06-05

Tested against the questions in `Questions for MCP.md`. Tools tested:
`search_papers`, `get_paper_by_doi`, `get_paper_fulltext`, `search_pis`, `get_pi`.

---

## Basic / Metadata

| Question | Result | Status |
|---|---|---|
| How many publications? | 956 papers | ✅ Works |
| Which groups work on electronic structure theory? | Rinke, Finley, Koblmüller, Müller-Caspary, Reuter | ✅ Works |
| Research focus of Prof Rinke? | Electronic Structure Theory Development; ML in Materials Science; Data Driven Materials Science | ✅ Works |
| Publications by PI in chronological order | Works for PIs with DOIs (34/42 have them); Rinke and ~8 others have 0 DOIs on their staff page | ⚠️ Partial |
| Paper with most co-authors from e-conversion | Best detection: 3 PI names in author list — limited by truncated author names in CSV | ⚠️ Weak |
| Papers in Nature journals | **NOT POSSIBLE** — no journal field in CSV or abstract cache | ❌ Missing data |
| Papers with two specific co-authors (e.g. Rinke + Scheffler) | `search_papers("Rinke Scheffler")` returns 5 results but matches on title/abstract, not author filter | ⚠️ Approximate only |

**Notes:**
- Author names in the CSV are truncated for non-ASCII surnames (known upstream scraper bug). This limits author-based filtering and co-authorship detection.
- Journal/venue metadata is absent from the current data pipeline. Would need OpenAlex `host_venue` to add it.

---

## Semantic / Content Understanding

| Question | Result | Status |
|---|---|---|
| Groups similar to Rinke by research topic | `search_pis("machine learning materials science electronic structure")` → Müller-Caspary, Reuter, Egger, Finley | ✅ Works (lexical) |
| Main open challenges across papers | `search_papers("open challenges limitations future work energy conversion")` returns relevant abstracts — but **synthesis across all 956 papers requires an LLM**, the tool only retrieves top 5 | ⚠️ Retrieval only |
| Papers on interfaces in energy conversion | Returns 5 relevant results (e.g. "Optical Metasurfaces for Energy Conversion", "Insights into Decoupled Solar Energy Conversion") | ✅ Works |
| ML applied to experimental data | Returns relevant papers (NMR simulations, self-driving labs, tensorial learning) | ✅ Works |

**Notes:**
- Semantic queries (synonyms, conceptual) are handled by BM25 — works when the vocabulary matches the abstract. Will miss papers that use different terminology. Adding vector embeddings would help.
- Questions requiring synthesis across the full corpus (Q9, Q12–Q17) are beyond what a single `search_papers` call can do — they need an LLM to iterate over many results and summarize. That's the "chat interface" use case.

---

## Cross-group / Interdisciplinary

| Question | Result | Status |
|---|---|---|
| Most complementary groups | Cannot answer directly — would need LLM to reason over retrieved PI profiles | ⚠️ Retrieval only |
| Topics in PI descriptions but few papers | Rough approach works: classify PIs as theory/experiment by keyword → 5 pure-theory, 5 pure-experiment PIs identified | ⚠️ Approximate |
| Experimental groups that co-published with theory groups | Not directly queryable — author names too truncated for reliable PI-to-paper linkage | ❌ Blocked by data |

---

## Temporal / Trend Analysis

| Question | Result | Status |
|---|---|---|
| ML-related paper share over years | Countable from title+abstract keywords: 2020:2, 2021:4, 2022:3, 2023:2, 2024:5, 2025:10 — clear upward trend | ✅ Works |
| Topics in early vs. recent papers | Doable with keyword queries filtered by year — but only top 5 results per query, not corpus-wide stats | ⚠️ Partial |
| Collaboration density over time | Not possible without a proper author-matching layer | ❌ Blocked by data |

---

## Adversarial / Graceful Failure

| Question | Result | Status |
|---|---|---|
| PI not in database ("Heisenberg") | Returns `{"error": "No PI found matching: Heisenberg"}` — clean failure | ✅ Correct |
| Absent topic ("topological insulators") | Returns 3 results — **turns out 1 paper genuinely is about topological insulators** (Gate-Tunable Helical Currents in Commensurate Topological Insulator/Graphene). The assumption this topic was absent was wrong. | ✅ Correct (surprising) |
| h-index | Not in data — must be flagged manually by the LLM using these tools | ❌ Missing data |
| Paper title with typo ("perovskyte effciency") | BM25 still returns relevant perovskite papers — typo tolerance is partial (tokenization helps when only some tokens are misspelled) | ✅ Reasonable |

---

## Graph / Network

| Question | Result | Status |
|---|---|---|
| Co-authorship graph of PIs | **Not directly possible** — PI DOI lists exist (34/42 PIs, 2,784 DOIs total) but author names in the paper records are truncated, preventing reliable PI-to-paper attribution | ❌ Blocked by data |
| Bridge PIs, community detection | Same blocker | ❌ Blocked by data |

**Fix path:** fetch full author lists from OpenAlex for the 956 DOIs and replace the truncated CSV entries. This is already listed in context.md's short-term tasks ("Fix author names").

---

## Summary

**Works well (retrieval):** named-PI lookup, PI keyword search, paper keyword search, basic temporal trend counting, graceful failure for unknown entities.

**Works partially:** author-based filtering, multi-token semantic queries, cross-referencing PI–paper linkage (34/42 PIs have DOI lists, 8 don't).

**Blocked by data gaps:**
1. **Truncated author names** — prevents co-authorship analysis and PI-to-paper attribution for graph questions.
2. **No journal/venue field** — cannot answer "Nature papers", "PRB papers" etc.
3. **No citation metrics** — cannot answer h-index or citation count queries.

**Blocked by architecture (retrieval-only):** synthesis questions (Q9, Q12–Q17) need an LLM layer on top of the tools. This is exactly what the planned chat interface will provide.

**Highest-priority fix:** add `journal` field (from OpenAlex `host_venue`) and fix truncated author names (from OpenAlex `authorships`). Both can be added to `build_abstracts_cache.py` in one pass.
