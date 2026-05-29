# Plan: Multi-Source Full Text Fetching Pipeline (v2)

## Context

V1 of `build_fulltext_cache.py` uses only Unpaywall and yielded ~37–40% success on a 30-DOI smoke test. Failures clustered around:

1. **Wiley/ACS bot-blocking** of publisher PDF URLs (403 even with browser UA)
2. **Landing-page HTML extractions** — Unpaywall sometimes returns the article landing URL, which only contains the abstract + UI text. Trafilatura correctly extracts what's there; the page just doesn't have the article body.
3. **Papers Unpaywall marks "closed"** that have an arXiv preprint anyway

Empirical evidence: PRB paper `10.1103/physrevb.110.125202` — Unpaywall and OpenAlex both said `is_oa: False`, but Semantic Scholar exposed `externalIds.ArXiv = "2408.10412"`. Downloading `arxiv.org/pdf/2408.10412` produced 90,522 chars of clean markdown.

**Goal:** widen the funnel by adding OpenAlex, Semantic Scholar, and arXiv as PDF sources alongside Unpaywall. Drop the HTML fallback to avoid landing-page noise. Expected lift: +10–25% recovery (especially for physics-heavy papers via arXiv).

---

## Approach

Per DOI, **query all four APIs in full** (no early exit) to collect every candidate URL anyone knows about. Deduplicate. Then download PDFs in a strict priority order based on URL host. If every PDF fails, fall back to HTML extraction from publisher landing pages — the `MIN_CHARS=3000` floor already filters abstract-only landing pages, and our smoke test confirmed real full-text HTML pages (Nature OA, some ACS OA) contribute ~10% of successes.

### Source matrix

| Source | API field used | Notes |
|---|---|---|
| Unpaywall | `oa_locations[*].url_for_pdf` | **Filter out landing-page-only locations** (those without `url_for_pdf`) |
| OpenAlex | `best_oa_location.pdf_url`, `locations[*].pdf_url` | Same DOI-based lookup as in `build_abstracts_cache.py:fetch_openalex` |
| Semantic Scholar | `openAccessPdf.url`, `externalIds.ArXiv` | Returns arXiv ID even when `openAccessPdf` is empty — the key insight |
| arXiv | `https://arxiv.org/pdf/{arxiv_id}` | Derived from arXiv IDs found in OpenAlex (`ids.arxiv`) or S2 (`externalIds.ArXiv`) |

### Priority order — exact mechanism

After collecting all PDF URLs from the four APIs and deduplicating, classify each URL by its hostname and sort by tier:

```python
from urllib.parse import urlparse

REPO_HOSTS = (
    "pmc.ncbi.nlm.nih.gov", "europepmc.org",
    "hal.", "eprints.", "repository.", "pure.",
    "edoc.", "elib.", "openresearch.", "osti.gov",
)

def host_tier(url: str) -> int:
    host = urlparse(url).netloc.lower()
    if "arxiv.org" in host:
        return 0  # arXiv — direct PDFs, no bot-blocking
    if any(r in host for r in REPO_HOSTS):
        return 1  # institutional/subject repositories
    return 2      # publisher (variable: Nature/Science usually OK, Wiley/ACS often 403)

pdf_urls.sort(key=host_tier)  # stable sort preserves source order within a tier
```

Try URLs sequentially; first success wins (return immediately, don't try the rest).

**Why this order:**
- arXiv: zero blocking, fastest, highest-quality preprint PDFs
- Repos: PMC/institutional repos host versions even when the publisher PDF is paywalled. Some have interstitials but many work
- Publishers: works for OA-friendly journals (Nature, Science, RSC, AIP); will fail for Wiley/ACS — but we've already exhausted better options by this point

**HTML fallback (after all PDFs fail):**
Collect HTML landing-page URLs from Unpaywall `oa_locations[*].url` (when no PDF was given), try each via trafilatura. `MIN_CHARS=3000` rejects landing pages with only abstracts. Publishers like Nature inline the full body in HTML — this recovers ~10% of corpus.

---

## Files to modify

| File | Change |
|---|---|
| `build_fulltext_cache.py` | Rewrite `try_locations()`; add `fetch_openalex_pdfs`, `fetch_semantic_scholar_pdfs`, `collect_urls`, `host_tier`. **Keep `extract_html`** as a final fallback (after all PDFs fail). |
| `requirements.txt` | Add `trafilatura` (still useful elsewhere) and `pymupdf4llm` if not already listed |
| `context.md` | Note: multi-source pipeline; arXiv recovers physics papers Unpaywall misses |

`server.py` and `fulltext_cache.json` schema stay the same.

---

## `build_fulltext_cache.py` — Key Functions

```python
def fetch_unpaywall(doi) -> tuple[list[str], list[str]]:
    """Return (pdf_urls, html_urls) from Unpaywall.
       PDFs: oa_locations[*].url_for_pdf (landing-page-only locations dropped)
       HTMLs: oa_locations[*].url where url_for_pdf is None and url doesn't end in .pdf
    """

def fetch_openalex(doi) -> tuple[list[str], str | None]:
    """Return (pdf_urls, arxiv_id).
       pdf_urls = [loc["pdf_url"] for loc in locations if loc.get("pdf_url")]
       arxiv_id from ids.arxiv if present
    """

def fetch_semantic_scholar(doi) -> tuple[str | None, str | None]:
    """Return (openAccessPdf.url, externalIds.ArXiv).
       Keep S2 even when openAccessPdf is empty — externalIds.ArXiv is the win.
    """

def collect_urls(doi) -> tuple[list[str], list[str]]:
    """Query all four sources, dedupe, sort PDFs by host tier.
       Returns (sorted_pdf_urls, html_urls).
    """
    pdf_urls, html_urls = [], []
    arxiv_ids = set()

    # 1. Unpaywall
    up_pdfs, up_htmls = fetch_unpaywall(doi)
    pdf_urls += up_pdfs
    html_urls += up_htmls

    # 2. OpenAlex
    oa_pdfs, oa_arxiv = fetch_openalex(doi)
    pdf_urls += oa_pdfs
    if oa_arxiv: arxiv_ids.add(oa_arxiv)

    # 3. Semantic Scholar
    s2_pdf, s2_arxiv = fetch_semantic_scholar(doi)
    if s2_pdf: pdf_urls.append(s2_pdf)
    if s2_arxiv: arxiv_ids.add(s2_arxiv)

    # 4. arXiv (derived from IDs found in 2/3)
    for aid in arxiv_ids:
        pdf_urls.append(f"https://arxiv.org/pdf/{aid}")

    # Dedupe (preserve first occurrence) then sort by host tier
    pdf_urls = list(dict.fromkeys(pdf_urls))
    pdf_urls.sort(key=host_tier)
    html_urls = list(dict.fromkeys(html_urls))

    return pdf_urls, html_urls

def fetch_fulltext(doi) -> dict | None:
    pdf_urls, html_urls = collect_urls(doi)

    # Tier 0/1/2: PDFs in host-tier order
    for url in pdf_urls:
        text = extract_pdf(url)
        if text:
            return {"fulltext": text, "url": url,
                    "source": "pdf", "source_origin": _origin(url)}

    # Tier 3: HTML landing-pages (MIN_CHARS filter removes abstract-only pages)
    for url in html_urls:
        text = extract_html(url)
        if text:
            return {"fulltext": text, "url": url,
                    "source": "html", "source_origin": _origin(url)}

    return None
```

`extract_pdf()`, `extract_html()`, `MIN_CHARS=3000`, browser UA, %PDF magic-byte check, incremental cache writing every 50 papers — all keep current behavior.

### Rate-limiting / politeness

- 0.3s sleep between any API call
- ~4 API calls + ≤4 PDF downloads per DOI worst case
- Estimated full run: 90–150 min for 956 DOIs

### Source attribution

Store which source contributed the winning URL in the cache:
```json
{
  "10.1103/...": {
    "fulltext": "...",
    "source": "pdf",
    "source_origin": "arxiv",  // new: arxiv | unpaywall | openalex | semantic_scholar
    "url": "https://arxiv.org/pdf/2408.10412",
    "char_count": 90522,
    "fetched_at": "2026-05-21"
  }
}
```

This makes it easy to audit later which source mattered most.

---

## Existing code to reuse

- `build_abstracts_cache.py:fetch_openalex` — same OpenAlex lookup pattern (replace inverted-index parsing with `pdf_url` / `ids.arxiv` extraction)
- `build_abstracts_cache.py:fetch_semantic_scholar` — same endpoint, change `fields` param to `openAccessPdf,externalIds`
- Existing `extract_pdf()`, `USER_AGENT`, `MIN_CHARS`, incremental-cache write logic — keep as-is

---

## Verification

1. Smoke test: rerun the 30-DOI random sample (`seed=42`) → expect ≥45% recovery (vs. 40% in v1), with several new hits coming from arXiv.
2. Confirm the PRB paper `10.1103/physrevb.110.125202` is now found via arXiv (not found in v1).
3. Inspect `source_origin` distribution after full run — gives a real picture of which source actually contributed.
4. Start the MCP server; call `get_paper_fulltext("10.1103/physrevb.110.125202")`; confirm body is returned.
5. Final coverage target: 45–55% of 956 (~430–520 papers) with clean markdown bodies.
