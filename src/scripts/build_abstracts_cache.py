"""Build abstracts_cache.json: abstracts plus OpenAlex metadata, one entry per DOI.

Abstract sources in priority order:
  1. e-conversion-Converted.enl (local SQLite, highest quality)
  2. OpenAlex API (fallback for DOIs not in .enl)
  3. Semantic Scholar API (second fallback)

OpenAlex is queried once per DOI regardless of where the abstract came from,
because it also supplies metadata the CSV lacks or mangles:
  - authors: full UTF-8 names (the CSV's author column is truncated for
    non-ASCII surnames — upstream scraper bug)
  - journal: from primary_location.source (host_venue as legacy fallback)
  - citation_count: cited_by_count, refreshed on every rebuild

Entry schema:
  {"abstract": str ("" if none found), "source": "enl|openalex|semantic_scholar|none",
   "authors": [str], "journal": str|None, "citation_count": int|None}

Full rebuild on every run (~956 OpenAlex calls, 3-5 min). Checkpoints every
50 DOIs so an interrupted run can be inspected, though reruns start fresh.
"""
import csv
import json
import re
import sqlite3
import time
import requests
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
ENL = DATA_DIR / "e-conversion-Converted.enl"
CSV = DATA_DIR / "data_publication_dois.csv"
OUTPUT = DATA_DIR / "abstracts_cache.json"
EMAIL = "audit@econversion.de"
CHECKPOINT_EVERY = 50

DOI_RE = re.compile(r"10\.\d{4,}/\S+", re.IGNORECASE)


def extract_doi(text):
    m = DOI_RE.search(text or "")
    return m.group(0).rstrip(".,;)>\"'").lower() if m else None


def load_enl_abstracts():
    con = sqlite3.connect(ENL)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT url, electronic_resource_number, abstract FROM enl_refs"
    ).fetchall()
    con.close()
    abstracts = {}
    for row in rows:
        doi = extract_doi(row["url"]) or extract_doi(row["electronic_resource_number"])
        if doi and row["abstract"] and row["abstract"].strip():
            abstracts[doi] = row["abstract"].strip()
    return abstracts


def fetch_openalex(doi):
    """Return {"abstract", "authors", "journal", "citation_count"} or None on failure."""
    try:
        r = requests.get(
            f"https://api.openalex.org/works/https://doi.org/{doi}",
            headers={"User-Agent": f"mailto:{EMAIL}"},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None

    abstract = None
    idx = data.get("abstract_inverted_index")
    if idx:
        positions = {}
        for word, pos_list in idx.items():
            for p in pos_list:
                positions[p] = word
        abstract = " ".join(positions[i] for i in sorted(positions))

    authors = [
        a["author"]["display_name"]
        for a in data.get("authorships") or []
        if a.get("author", {}).get("display_name")
    ]

    # primary_location.source supersedes host_venue in the current OpenAlex schema.
    journal = None
    src = (data.get("primary_location") or {}).get("source") or {}
    if src.get("display_name"):
        journal = src["display_name"]
    elif (data.get("host_venue") or {}).get("display_name"):
        journal = data["host_venue"]["display_name"]

    return {
        "abstract": abstract,
        "authors": authors,
        "journal": journal,
        "citation_count": data.get("cited_by_count"),
    }


def fetch_semantic_scholar(doi):
    try:
        r = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
            params={"fields": "abstract"},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        return r.json().get("abstract")
    except Exception:
        return None


def main():
    with open(CSV, encoding="utf-8") as f:
        csv_dois = sorted({row["article_doi"].lower() for row in csv.DictReader(f) if row["article_doi"]})

    print("Loading abstracts from .enl...")
    enl = load_enl_abstracts()
    print(f"  ENL: {sum(1 for d in csv_dois if d in enl)}/{len(csv_dois)} abstracts covered")
    print(f"Fetching OpenAlex metadata for all {len(csv_dois)} DOIs...\n")

    cache = {}
    for i, doi in enumerate(csv_dois, 1):
        oa = fetch_openalex(doi)

        if doi in enl:
            abstract, source = enl[doi], "enl"
        elif oa and oa["abstract"] and len(oa["abstract"]) >= 50:
            abstract, source = oa["abstract"], "openalex"
        else:
            abstract = fetch_semantic_scholar(doi)
            source = "semantic_scholar"
            if not abstract or len(abstract) < 50:
                abstract, source = "", "none"

        cache[doi] = {
            "abstract": abstract,
            "source": source,
            "authors": oa["authors"] if oa else [],
            "journal": oa["journal"] if oa else None,
            "citation_count": oa["citation_count"] if oa else None,
        }

        tag = f"[{source}]" + ("" if oa else " [no openalex]")
        print(f"  [{i:4d}/{len(csv_dois)}] {doi[:45]:<45}  {tag}")

        if i % CHECKPOINT_EVERY == 0:
            with open(OUTPUT, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        time.sleep(0.15)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    sources = {}
    for v in cache.values():
        sources[v["source"]] = sources.get(v["source"], 0) + 1
    with_abstract = sum(1 for v in cache.values() if v["abstract"])
    with_authors = sum(1 for v in cache.values() if v["authors"])
    with_journal = sum(1 for v in cache.values() if v["journal"])

    print(f"\nCache written to {OUTPUT.name}")
    print(f"Abstracts: {with_abstract}/{len(cache)} ({100*with_abstract/len(cache):.1f}%)")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")
    print(f"Authors:  {with_authors}/{len(cache)}")
    print(f"Journal:  {with_journal}/{len(cache)}")


if __name__ == "__main__":
    main()
