"""Ingest locally-supplied full-text PDFs (a collaborator's local full-texts) into fulltext_cache.json.

Unlike build_fulltext_cache.py, which fetches PDFs from OA sources over the network,
this reads PDFs already sitting in data/pdfs/. Filenames encode the DOI with assorted
separator spellings ('10.1002_adfm.201900233.pdf', '10_1002_adfm_202505935.pdf'), so
each file is matched to a cluster DOI by a separator-agnostic fingerprint rather than
by reversing one fixed mangling.

Handled quirks, all found by double-checking the folder before ingest:
  - Files prefixed 'duplicate_' are redundant copies the depositor already flagged; skipped.
  - A few files are HTML landing pages saved with a .pdf name (header '<!DOC', not '%PDF').
    Real PDFs are extracted with pymupdf4llm; HTML files fall back to trafilatura, which
    recovers the body when the publisher inlines it (e.g. Nature Reviews) and is rejected
    by the MIN_CHARS floor when the page is only an abstract/figure-viewer stub.
  - When one DOI has several files, a real %PDF is preferred over an HTML copy.

Cache entries are keyed by lowercase DOI to match search.py / get_paper_fulltext lookups,
tagged source_origin="collaborator" so this provenance is auditable alongside arxiv/pmc/publisher.
Existing entries are never overwritten (this folder is complementary to the fetched cache);
an unexpected overlap is reported loudly rather than silently clobbering.
"""
import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

import pymupdf4llm
import trafilatura

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CSV = DATA_DIR / "data_publication_dois.csv"
PDF_DIR = DATA_DIR / "pdfs"
OUTPUT = DATA_DIR / "fulltext_cache.json"
MIN_CHARS = 3000  # same floor as build_fulltext_cache.py: rejects landing-page stubs
SOURCE_ORIGIN = "collaborator"


def fingerprint(doi: str) -> str:
    """Separator-agnostic DOI key: drop _ / . - and lowercase, so every filename
    spelling of the same DOI collapses to one comparable token."""
    return re.sub(r"[_/.\s-]", "", doi.lower())


def load_corpus() -> dict:
    """fingerprint -> canonical lowercase DOI, for the 956 cluster publications."""
    corpus = {}
    with open(CSV, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            doi = row.get("article_doi", "").strip()
            if doi:
                corpus[fingerprint(doi)] = doi.lower()
    return corpus


def is_real_pdf(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(5).startswith(b"%PDF")


def extract(path: Path) -> tuple[str | None, str | None]:
    """Return (text, source) where source is 'pdf' or 'html'. None text = unusable."""
    if is_real_pdf(path):
        try:
            text = pymupdf4llm.to_markdown(str(path))
        except Exception as exc:  # corrupt PDF slips past the header check
            print(f"    ! pymupdf4llm failed on {path.name}: {exc}", flush=True)
            return None, None
        return (text if text and len(text) >= MIN_CHARS else None), "pdf"
    # Not a PDF: an HTML page saved with a .pdf name. Recover the body if present.
    html = path.read_text(encoding="utf-8", errors="ignore")
    text = trafilatura.extract(html, output_format="markdown", include_tables=False)
    return (text if text and len(text) >= MIN_CHARS else None), "html"


def main() -> None:
    corpus = load_corpus()

    # Group corpus-matching PDF files by DOI; note the strays for the report.
    by_doi: dict[str, list[Path]] = {}
    duplicates = 0
    non_corpus: list[str] = []
    for path in sorted(PDF_DIR.glob("*.pdf")):
        if path.name.lower().startswith("duplicate_"):
            duplicates += 1
            continue
        doi = corpus.get(fingerprint(path.stem))
        if doi is None:
            non_corpus.append(path.name)
            continue
        by_doi.setdefault(doi, []).append(path)

    cache = {}
    if OUTPUT.exists():
        with open(OUTPUT, encoding="utf-8") as fh:
            cache = json.load(fh)

    already = [d for d in by_doi if d in cache]
    to_ingest = [d for d in by_doi if d not in cache]

    print(f"PDF folder            : {PDF_DIR}")
    print(f"Corpus DOIs in folder : {len(by_doi)}")
    print(f"  already in cache    : {len(already)} (skipped, not overwritten)")
    print(f"  to ingest           : {len(to_ingest)}")
    print(f"duplicate_ files      : {duplicates} (skipped)")
    print(f"non-corpus files      : {len(non_corpus)} (skipped)")
    print(f"existing cache size   : {len(cache)}\n", flush=True)

    today = str(date.today())
    counts = {"pdf": 0, "html": 0, "rejected": 0}
    rejected: list[str] = []

    for i, doi in enumerate(sorted(to_ingest), 1):
        # Prefer a genuine %PDF over an HTML-as-pdf copy when a DOI has several files.
        files = sorted(by_doi[doi], key=lambda p: (not is_real_pdf(p), p.name))
        chosen = files[0]
        text, source = extract(chosen)
        if text is None:
            counts["rejected"] += 1
            rejected.append(doi)
            print(f"  [{i:3d}/{len(to_ingest)}] {doi[:48]:<48}  [REJECTED {chosen.name}]", flush=True)
            continue
        cache[doi] = {
            "fulltext": text,
            "source": source,
            "source_origin": SOURCE_ORIGIN,
            "url": f"local:data/pdfs/{chosen.name}",
            "char_count": len(text),
            "fetched_at": today,
        }
        counts[source] += 1
        print(f"  [{i:3d}/{len(to_ingest)}] {doi[:48]:<48}  [{source}] {len(text)} chars", flush=True)

        if i % 50 == 0:
            with open(OUTPUT, "w", encoding="utf-8") as fh:
                json.dump(cache, fh, ensure_ascii=False, indent=2)

    with open(OUTPUT, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, indent=2)

    print(f"\nIngested: pdf={counts['pdf']}  html={counts['html']}  rejected={counts['rejected']}")
    if rejected:
        print("Rejected (no body above MIN_CHARS - left as missing full text):")
        for d in rejected:
            print(f"  {d}")
    print(f"\nCache written to {OUTPUT} - now {len(cache)} entries.")
    if non_corpus:
        print(f"\n{len(non_corpus)} non-corpus files were NOT ingested (not on the 956-paper list):")
        for name in sorted(non_corpus):
            print(f"  {name}")


if __name__ == "__main__":
    sys.exit(main())
