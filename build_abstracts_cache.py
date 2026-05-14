"""Build abstracts_cache.json from local .enl (primary) with API fallback for missing DOIs.

Sources in priority order:
  1. e-conversion-Converted.enl (local SQLite, highest quality)
  2. OpenAlex API (fallback for DOIs not in .enl)
  3. Semantic Scholar API (second fallback)
"""
import csv
import json
import re
import sqlite3
import time
import requests
from pathlib import Path

BASE = Path(__file__).parent
ENL = BASE / "e-conversion-Converted.enl"
CSV = BASE / "data_publication_dois.csv"
OUTPUT = BASE / "abstracts_cache.json"
EMAIL = "audit@econversion.de"

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
    try:
        r = requests.get(
            f"https://api.openalex.org/works/https://doi.org/{doi}",
            headers={"User-Agent": f"mailto:{EMAIL}"},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        idx = r.json().get("abstract_inverted_index")
        if not idx:
            return None
        positions = {}
        for word, pos_list in idx.items():
            for p in pos_list:
                positions[p] = word
        return " ".join(positions[i] for i in sorted(positions))
    except Exception:
        return None


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

    print(f"Loading abstracts from .enl...")
    enl = load_enl_abstracts()

    cache = {}
    missing = []

    for doi in csv_dois:
        if doi in enl:
            cache[doi] = {"abstract": enl[doi], "source": "enl"}
        else:
            missing.append(doi)

    print(f"  ENL:     {len(cache)}/{len(csv_dois)} papers covered")
    print(f"  Missing: {len(missing)} -- fetching from APIs...\n")

    for i, doi in enumerate(missing, 1):
        abstract = fetch_openalex(doi)
        source = "openalex"
        if not abstract or len(abstract) < 50:
            abstract = fetch_semantic_scholar(doi)
            source = "semantic_scholar"
        if abstract and len(abstract) >= 50:
            cache[doi] = {"abstract": abstract, "source": source}
            print(f"  [{i:2d}/{len(missing)}] {doi[:45]:<45}  [{source}]")
        else:
            print(f"  [{i:2d}/{len(missing)}] {doi[:45]:<45}  [NOT FOUND]")
        time.sleep(0.2)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    sources = {}
    for v in cache.values():
        sources[v["source"]] = sources.get(v["source"], 0) + 1

    print(f"\nCache written to {OUTPUT.name}")
    print(f"Total: {len(cache)}/{len(csv_dois)} papers ({100*len(cache)/len(csv_dois):.1f}%)")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")


if __name__ == "__main__":
    main()