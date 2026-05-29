"""Build fulltext_cache.json by fetching full text for OA papers.

Multi-source pipeline: queries Unpaywall, OpenAlex, and Semantic Scholar to
collect every known PDF URL for each DOI, derives an arXiv URL if any source
exposes an arXiv ID, then deduplicates and tries them in host-priority order:

    0. arXiv          (no bot-blocking, fastest)
    1. Repositories   (PMC, institutional repos)
    2. Publishers     (Nature/Science/RSC usually OK; Wiley/ACS often 403)

PDF extraction via pymupdf4llm (markdown). If every PDF fails, falls back to
HTML extraction via trafilatura with a MIN_CHARS=3000 floor that rejects
abstract-only landing pages.

Run incrementally — DOIs already in the cache are skipped unless --force is passed.
"""
import csv
import json
import re
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
import requests
import trafilatura
import pymupdf4llm
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV = DATA_DIR / "data_publication_dois.csv"
OUTPUT = DATA_DIR / "fulltext_cache.json"
EMAIL = "audit@econversion.de"
MIN_CHARS = 3000  # full articles are typically 10k+ chars; this filters landing pages
FORCE = "--force" in sys.argv
API_SLEEP = 0.3  # politeness between API calls

# Many publishers (ACS, Wiley, PMC) reject requests without a browser-like UA.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    f"(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 (mailto:{EMAIL})"
)

REPO_HOST_MARKERS = (
    "pmc.ncbi.nlm.nih.gov", "europepmc.org",
    "hal.", "eprints.", "repository.", "pure.",
    "edoc.", "elib.", "openresearch.", "osti.gov",
)


def host_tier(url: str) -> int:
    """Sort key: 0=arXiv, 1=repository, 2=publisher."""
    host = urlparse(url).netloc.lower()
    if "arxiv.org" in host:
        return 0
    if any(m in host for m in REPO_HOST_MARKERS):
        return 1
    return 2


def _origin(url: str) -> str:
    """Short label for which kind of host produced the winning URL."""
    host = urlparse(url).netloc.lower()
    if "arxiv.org" in host:
        return "arxiv"
    if any(m in host for m in REPO_HOST_MARKERS):
        return "repository"
    return "publisher"


def fetch_unpaywall(doi):
    """Return (pdf_urls, html_urls) from Unpaywall, or ([], [])."""
    try:
        r = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": EMAIL},
            timeout=15,
        )
        if r.status_code != 200:
            return [], []
        locations = r.json().get("oa_locations") or []
    except Exception:
        return [], []

    pdf_urls, html_urls = [], []
    for loc in locations:
        pdf = loc.get("url_for_pdf")
        url = loc.get("url") or ""
        if pdf:
            pdf_urls.append(pdf)
        elif url.lower().endswith(".pdf"):
            pdf_urls.append(url)
        elif url:
            html_urls.append(url)
    return pdf_urls, html_urls


def fetch_openalex(doi):
    """Return (pdf_urls, arxiv_id_or_None, pmcid_or_None) from OpenAlex."""
    try:
        r = requests.get(
            f"https://api.openalex.org/works/https://doi.org/{doi}",
            headers={"User-Agent": f"mailto:{EMAIL}"},
            timeout=15,
        )
        if r.status_code != 200:
            return [], None, None
        data = r.json()
    except Exception:
        return [], None, None

    pdf_urls = []
    for loc in (data.get("locations") or []):
        pdf = loc.get("pdf_url")
        if pdf:
            pdf_urls.append(pdf)
    best = (data.get("best_oa_location") or {}).get("pdf_url")
    if best:
        pdf_urls.append(best)

    ids = data.get("ids") or {}
    arxiv_id = None
    arxiv_field = ids.get("arxiv")
    if arxiv_field:
        # Sometimes returned as a full URL, sometimes just the ID
        arxiv_id = arxiv_field.split("/")[-1].replace("arXiv:", "").strip()

    # OpenAlex exposes PMCID in ids.pmcid as a full URL like
    # "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567"
    pmcid = None
    pmcid_field = ids.get("pmcid")
    if pmcid_field:
        m = re.search(r"PMC\d+", pmcid_field)
        if m:
            pmcid = m.group(0)

    return pdf_urls, arxiv_id, pmcid


def fetch_semantic_scholar(doi):
    """Return (pdf_url_or_None, arxiv_id_or_None) from Semantic Scholar."""
    try:
        r = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
            params={"fields": "openAccessPdf,externalIds"},
            timeout=15,
        )
        if r.status_code != 200:
            return None, None
        data = r.json()
    except Exception:
        return None, None

    pdf = (data.get("openAccessPdf") or {}).get("url") or None
    arxiv_id = (data.get("externalIds") or {}).get("ArXiv")
    return pdf, arxiv_id


def fetch_arxiv_id_by_doi(doi):
    """Search arXiv's API for a preprint matching this DOI. Returns arxiv_id or None.

    Many authors cross-list the published DOI on their arXiv submission; arXiv's
    query API exposes that. This catches preprints that neither OpenAlex nor S2
    surface as `ids.arxiv` / `externalIds.ArXiv`.
    """
    try:
        r = requests.get(
            "http://export.arxiv.org/api/query",
            params={"search_query": f"doi:{doi}", "max_results": 1},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        # Atom feed: extract the first <id> after the feed-level <id>
        # entry id looks like "http://arxiv.org/abs/2301.12345v2"
        m = re.search(r"<entry>.*?<id>http://arxiv\.org/abs/([^<]+)</id>", r.text, re.DOTALL)
        if not m:
            return None
        # Strip version suffix; the bare ID works for /pdf/<id>
        return re.sub(r"v\d+$", "", m.group(1)).strip()
    except Exception:
        return None


def _jats_to_markdown(xml_text):
    """Render a JATS <article> XML body to plain markdown.

    Emits the article title (H1), abstract (H2), and section bodies with nested
    heading depth. Skips floats (figures, tables, formulas) — they don't carry
    over usefully into plain text and trafilatura-comparable output is the goal.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    article = root.find(".//article")
    if article is None:
        return None

    out = []
    title = article.findtext(".//front//article-title")
    if title:
        out.append(f"# {title.strip()}")
    for abs_el in article.findall(".//front//abstract"):
        out.append("\n## Abstract")
        out.append(_walk_jats(abs_el, depth=2))
    body = article.find("body")
    if body is not None:
        out.append(_walk_jats(body, depth=1))
    return "\n\n".join(p for p in out if p and p.strip())


def _walk_jats(el, depth=1):
    parts = []
    for child in el:
        tag = child.tag
        if tag == "sec":
            t = child.findtext("title")
            if t:
                parts.append(f'\n{"#" * min(depth + 1, 6)} {t.strip()}')
            parts.append(_walk_jats(child, depth=depth + 1))
        elif tag == "title":
            continue  # handled by parent sec
        elif tag == "p":
            parts.append("".join(child.itertext()).strip())
        elif tag in ("list", "disp-quote"):
            parts.append("".join(child.itertext()).strip())
        elif tag in ("fig", "table-wrap", "disp-formula"):
            continue
        else:
            parts.append(_walk_jats(child, depth=depth))
    return "\n\n".join(p for p in parts if p and p.strip())


def fetch_pmc(doi, pmcid=None):
    """Try to get full text from PubMed Central. Returns markdown or None.

    PMC holds NIH/HHMI-deposited author manuscripts even when the publisher
    paywalls the typeset version, so this rescues a meaningful fraction of
    ACS/Wiley/Nature papers that the URL pipeline can't reach.

    Discovery: uses the caller-supplied PMCID if known (from OpenAlex), else
    queries NCBI's idconv endpoint with the DOI.
    """
    if not pmcid:
        try:
            r = requests.get(
                "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
                params={
                    "ids": doi,
                    "format": "json",
                    "tool": "econversion",
                    "email": EMAIL,
                },
                timeout=15,
            )
            if r.status_code != 200:
                return None
            records = (r.json().get("records") or [])
            if not records or not records[0].get("pmcid"):
                return None
            pmcid = records[0]["pmcid"]
        except Exception:
            return None

    pmc_num = pmcid.replace("PMC", "").strip()
    try:
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={
                "db": "pmc",
                "id": pmc_num,
                "rettype": "full",
                "tool": "econversion",
                "email": EMAIL,
            },
            timeout=30,
        )
        if r.status_code != 200:
            return None
    except Exception:
        return None

    md = _jats_to_markdown(r.text)
    if md and len(md) >= MIN_CHARS:
        return {"fulltext": md, "pmcid": pmcid}
    return None


def _expand_urls(urls):
    """For URLs with query strings, also add the bare variant.
    Some publishers (e.g. science.org) reject ?download=true but serve the bare URL."""
    expanded = []
    for u in urls:
        expanded.append(u)
        if "?" in u:
            expanded.append(u.split("?", 1)[0])
    return expanded


def collect_urls(doi):
    """Query all sources. Returns (pdf_urls, html_urls, pmcid_or_None).

    pdf_urls are deduped and sorted by host tier (arXiv first, then repos,
    then publishers). The pmcid (if any) is surfaced separately so the caller
    can try PMC's structured-XML path between arXiv and the repo URLs.
    """
    pdf_urls, html_urls = [], []
    arxiv_ids = set()
    pmcid = None

    up_pdfs, up_htmls = fetch_unpaywall(doi)
    pdf_urls += up_pdfs
    html_urls += up_htmls
    time.sleep(API_SLEEP)

    oa_pdfs, oa_arxiv, oa_pmcid = fetch_openalex(doi)
    pdf_urls += oa_pdfs
    if oa_arxiv:
        arxiv_ids.add(oa_arxiv)
    if oa_pmcid:
        pmcid = oa_pmcid
    time.sleep(API_SLEEP)

    s2_pdf, s2_arxiv = fetch_semantic_scholar(doi)
    if s2_pdf:
        pdf_urls.append(s2_pdf)
    if s2_arxiv:
        arxiv_ids.add(s2_arxiv)
    time.sleep(API_SLEEP)

    # Last-resort arXiv lookup: query arXiv's own API by DOI for cases where
    # neither OpenAlex nor S2 surfaced an arXiv ID. Cheap (one request) and
    # only fires when the other sources came up empty.
    if not arxiv_ids:
        arxiv_id = fetch_arxiv_id_by_doi(doi)
        if arxiv_id:
            arxiv_ids.add(arxiv_id)
        time.sleep(API_SLEEP)

    for aid in arxiv_ids:
        pdf_urls.append(f"https://arxiv.org/pdf/{aid}")

    pdf_urls = _expand_urls(pdf_urls)
    pdf_urls = list(dict.fromkeys(pdf_urls))
    pdf_urls.sort(key=host_tier)
    html_urls = list(dict.fromkeys(html_urls))
    return pdf_urls, html_urls, pmcid


def extract_pdf(url):
    """Download PDF and extract markdown text via pymupdf4llm."""
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT}, allow_redirects=True)
        if r.status_code != 200:
            return None
        # Some hosts (PMC interstitial, Wiley 403 page) return HTML instead of a PDF
        if not r.content.startswith(b"%PDF"):
            return None
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(r.content)
            tmp_path = f.name
        try:
            text = pymupdf4llm.to_markdown(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        if text and len(text) >= MIN_CHARS:
            return text
        return None
    except Exception:
        return None


def extract_html(url):
    """Fetch and extract main article text from an HTML URL via trafilatura."""
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT}, allow_redirects=True)
        if r.status_code != 200:
            return None
        text = trafilatura.extract(r.text, output_format="markdown", include_tables=False)
        if text and len(text) >= MIN_CHARS:
            return text
        return None
    except Exception:
        return None


def fetch_fulltext(doi):
    """Try every candidate URL in priority order. Returns dict with text/url/source or None.

    Order: arXiv PDFs (tier 0) → PMC structured XML (tier 0.5) → repo PDFs
    (tier 1) → publisher PDFs (tier 2) → HTML fallback.
    """
    pdf_urls, html_urls, pmcid = collect_urls(doi)

    # Tier 0: arXiv PDFs first (clean, no bot-blocking).
    arxiv_urls = [u for u in pdf_urls if host_tier(u) == 0]
    rest_urls = [u for u in pdf_urls if host_tier(u) != 0]
    for url in arxiv_urls:
        text = extract_pdf(url)
        if text:
            return {
                "fulltext": text,
                "source": "pdf",
                "source_origin": _origin(url),
                "url": url,
            }

    # Tier 0.5: PMC eutils. Structured JATS XML, often rescues NIH-funded
    # papers that publishers (ACS/Wiley/Nature) paywall.
    pmc_result = fetch_pmc(doi, pmcid=pmcid)
    if pmc_result:
        return {
            "fulltext": pmc_result["fulltext"],
            "source": "pmc",
            "source_origin": "pmc",
            "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_result['pmcid']}/",
        }

    # Tiers 1 and 2: repository then publisher PDFs (already sorted by host_tier).
    for url in rest_urls:
        text = extract_pdf(url)
        if text:
            return {
                "fulltext": text,
                "source": "pdf",
                "source_origin": _origin(url),
                "url": url,
            }

    for url in html_urls:
        text = extract_html(url)
        if text:
            return {
                "fulltext": text,
                "source": "html",
                "source_origin": _origin(url),
                "url": url,
            }

    return None


def main():
    import random
    with open(CSV, encoding="utf-8") as f:
        dois = sorted({row["article_doi"].lower() for row in csv.DictReader(f) if row["article_doi"]})
    # Shuffle deterministically so we don't slam the same publisher 50× in a row
    # (which can trigger per-host throttling) and so the success rate during the
    # run reflects the corpus average rather than alphabetical bias.
    random.Random(42).shuffle(dois)

    cache = {}
    if OUTPUT.exists() and not FORCE:
        with open(OUTPUT, encoding="utf-8") as f:
            cache = json.load(f)

    to_fetch = [doi for doi in dois if doi not in cache]

    print(f"Total DOIs:     {len(dois)}")
    print(f"Already cached: {len(cache)}")
    print(f"To fetch:       {len(to_fetch)}\n")

    today = str(date.today())
    counts = {"pdf": 0, "html": 0, "pmc": 0, "not_found": 0}
    origins = {"arxiv": 0, "pmc": 0, "repository": 0, "publisher": 0}

    for i, doi in enumerate(to_fetch, 1):
        result = fetch_fulltext(doi)
        if result:
            cache[doi] = {
                **result,
                "char_count": len(result["fulltext"]),
                "fetched_at": today,
            }
            counts[result["source"]] += 1
            origins[result["source_origin"]] += 1
            label = f"[{result['source']}/{result['source_origin']}] {len(result['fulltext'])} chars"
        else:
            counts["not_found"] += 1
            label = "[NOT FOUND]"

        print(f"  [{i:3d}/{len(to_fetch)}] {doi[:48]:<48}  {label}")

        if i % 50 == 0:
            with open(OUTPUT, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    found = counts["pdf"] + counts["html"] + counts["pmc"]
    print(f"\nCache written to {OUTPUT.name}")
    print(f"Total: {len(cache)}/{len(dois)} papers ({100*len(cache)/len(dois):.1f}%)")
    print(f"  pdf:       {counts['pdf']}")
    print(f"  html:      {counts['html']}")
    print(f"  pmc:       {counts['pmc']}")
    print(f"  not found: {counts['not_found']}")
    print(f"Origin of winning URL:")
    print(f"  arxiv:      {origins['arxiv']}")
    print(f"  pmc:        {origins['pmc']}")
    print(f"  repository: {origins['repository']}")
    print(f"  publisher:  {origins['publisher']}")
    if to_fetch:
        print(f"Success rate this run: {100*found/len(to_fetch):.1f}%")


if __name__ == "__main__":
    main()
