"""Scrape the full e-conversion publication list from the website.

The corpus (`data/sources/data_publication_dois.csv` + `scraper_cache.json`) was
originally produced by an external tool. This makes that step reproducible: it
walks every page of the teachpress list at e-conversion.de/publikationen and
parses the BibTeX record each entry exposes.

Pagination: teachpress uses `?limit=<page>` (1-based, 50 entries/page); the walk
stops at the first page with no entries. Each entry is a full BibTeX block with
title / author / doi / year / journal / abstract / issn / volume / pages, so we
capture the bibliographic metadata, not just the DOI. (The `url` field is a Web
of Science accession, not a full-text link, so full text still comes from
build_fulltext_cache.py / ingest_local_pdfs.py.)

Server-side rendered HTML, parsed with regex to stay dependency-free, matching
build_pis_cache.py.

Outputs:
  - data/sources/scraper_cache.json   (always) — the raw enriched scrape.
  - data/sources/data_publication_dois.csv (only with --write-csv) — the runtime
    corpus. Regenerating it changes the paper set the app loads and means the
    abstracts / embeddings / fulltext / graph caches should be rebuilt, so it is
    opt-in. Existing dataset_* / enl_url columns are merged back in by DOI.

Usage:
    python src/scripts/build_publications_list.py            # scrape + report delta
    python src/scripts/build_publications_list.py --write-csv  # also update the corpus CSV
"""
import csv
import html as htmllib
import io
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CSV_PATH = DATA_DIR / "sources" / "data_publication_dois.csv"
SCRAPE_PATH = DATA_DIR / "sources" / "scraper_cache.json"
LISTING_URL_TMPL = (
    "https://www.e-conversion.de/de/publikationen/"
    "?limit={page}&tgid=&yr=&type=&usr=&auth=&tsr="
)
MAX_PAGES = 60  # safety cap; the list is ~31 pages
SLEEP = 0.4
WRITE_CSV = "--write-csv" in sys.argv

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 "
    "(mailto:audit@econversion.de)"
)
HEADERS = {"User-Agent": USER_AGENT}

# LaTeX accent command -> combining diacritic, applied to the following letter.
_ACCENTS = {
    "'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308", "~": "\u0303",
    "=": "\u0304", ".": "\u0307", "u": "\u0306", "v": "\u030C", "H": "\u030B",
    "c": "\u0327", "k": "\u0328", "r": "\u030A",
}
# Standalone LaTeX letter commands.
_LIGATURES = {
    r"\ss": "\u00DF", r"\o": "\u00F8", r"\O": "\u00D8", r"\aa": "\u00E5",
    r"\AA": "\u00C5", r"\ae": "\u00E6", r"\AE": "\u00C6", r"\l": "\u0142",
    r"\L": "\u0141", r"\i": "i", r"\j": "j",
}


def _delatex(s: str) -> str:
    """Turn BibTeX/LaTeX-escaped text into plain Unicode."""
    # \'{e}, \"{u}, \c{c}  and the braceless \'e forms -> accented letter
    def accent(m):
        acc, letter = m.group(1), m.group(2)
        comb = _ACCENTS.get(acc)
        return __import__("unicodedata").normalize("NFC", letter + comb) if comb else letter
    s = re.sub(r"\\([`'^\"~=.uvHckr])\{\\?([a-zA-Z])\}", accent, s)
    s = re.sub(r"\\([`'^\"~=.])\s*([a-zA-Z])", accent, s)
    for cmd, ch in _LIGATURES.items():
        s = s.replace(cmd + "{}", ch).replace(cmd + " ", ch + " ").replace(cmd, ch)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\\&", "&", s)
    s = htmllib.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def fetch(page: int) -> str:
    r = requests.get(LISTING_URL_TMPL.format(page=page), headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


def parse_entries(html: str) -> list[dict]:
    """One dict per BibTeX block on the page. Fields are on `<br />`-separated
    lines as `name = {value}`; the value can itself contain braces (\\'{e}), so
    we take everything up to the field terminator, not the first brace."""
    records = []
    for block in re.finditer(r"@(\w+)\{[^,]*,\s*<br\s*/?>(.*?)<br\s*/?>\s*\}", html, re.S):
        body = block.group(2)
        fields = {}
        for line in re.split(r"<br\s*/?>", body):
            m = re.match(r"\s*(\w+)\s*=\s*\{(.*)\}\s*,?\s*$", line, re.S)
            if m:
                fields[m.group(1).lower()] = _delatex(m.group(2))
        doi = re.sub(r"[.,;)\]]+$", "", fields.get("doi", "")).lower()
        if not doi:
            continue
        records.append({
            "article_doi": doi,
            "title": fields.get("title", ""),
            "authors": fields.get("author", ""),
            "year": fields.get("year", ""),
            "journal": fields.get("journal", ""),
            "abstract": fields.get("abstract", ""),
            "issn": fields.get("issn", ""),
            "volume": fields.get("volume", ""),
            "pages": fields.get("pages", ""),
            "type": block.group(1).lower(),
        })
    return records


def scrape_all() -> list[dict]:
    seen, out = set(), []
    for page in range(1, MAX_PAGES + 1):
        recs = parse_entries(fetch(page))
        if not recs:
            print(f"page {page:>2}: empty — stopping")
            break
        new = [r for r in recs if r["article_doi"] not in seen]
        seen.update(r["article_doi"] for r in new)
        out.extend(new)
        print(f"page {page:>2}: {len(recs):>2} entries ({len(new)} new, {len(out)} total)")
        time.sleep(SLEEP)
    return out


def current_corpus_dois() -> set:
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH, encoding="utf-8") as fh:
        return {(row.get("article_doi") or "").strip().lower()
                for row in csv.DictReader(fh) if (row.get("article_doi") or "").strip()}


def write_csv(records: list[dict]) -> None:
    """Regenerate the corpus CSV, merging dataset_* / enl_url from the old CSV by DOI."""
    old = {}
    if CSV_PATH.exists():
        with open(CSV_PATH, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                old[(row.get("article_doi") or "").strip().lower()] = row
    cols = ["article_doi", "article_title", "article_authors", "article_year",
            "enl_url", "dataset_doi", "dataset_title", "dataset_type"]
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in records:
            prev = old.get(r["article_doi"], {})
            w.writerow({
                "article_doi": r["article_doi"],
                "article_title": r["title"],
                "article_authors": r["authors"],
                "article_year": r["year"],
                "enl_url": prev.get("enl_url", ""),
                "dataset_doi": prev.get("dataset_doi", ""),
                "dataset_title": prev.get("dataset_title", ""),
                "dataset_type": prev.get("dataset_type", ""),
            })


def main() -> None:
    before = current_corpus_dois()
    records = scrape_all()
    scraped = {r["article_doi"] for r in records}

    SCRAPE_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    new = scraped - before
    gone = before - scraped
    with_abstract = sum(1 for r in records if r["abstract"])
    print(f"\nScraped {len(records)} publications ({len(scraped)} unique DOIs) on {date.today()}")
    print(f"  with abstract : {with_abstract}")
    print(f"  wrote scrape  : {SCRAPE_PATH}")
    print(f"\nDelta vs current corpus ({len(before)} DOIs):")
    print(f"  new on site (not in corpus) : {len(new)}")
    print(f"  in corpus but not on site   : {len(gone)}")
    if gone:
        for d in sorted(gone)[:20]:
            print(f"      - {d}")
        if len(gone) > 20:
            print(f"      … +{len(gone) - 20} more")

    if WRITE_CSV:
        write_csv(records)
        print(f"\n--write-csv: regenerated {CSV_PATH} with {len(records)} rows.")
        print("  Downstream caches now need rebuilding for the new DOIs:")
        print("    python build.py            # abstracts, embeddings, fulltext, graph")
    else:
        print("\n(report only — pass --write-csv to update the corpus CSV, then rebuild caches)")


if __name__ == "__main__":
    main()
