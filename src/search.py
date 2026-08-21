import csv
import json
from pathlib import Path
from rank_bm25 import BM25Okapi

from logging_config import get_logger

log = get_logger("search")

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_CACHE_PATH = _DATA_DIR / "cache" / "abstracts_cache.json"


def safe_load_json(path: Path, label: str):
    """Load a JSON cache, returning None (not raising) on absence or corruption.

    A missing or malformed optional cache should degrade one feature, not crash
    the whole server at import. Callers substitute an empty default and the
    status tool reports the gap.
    """
    if not path.exists():
        log.warning("cache missing", extra={"fields": {"cache": label, "path": str(path)}})
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        log.error("cache corrupt", exc_info=True, extra={"fields": {"cache": label}})
        return None


_ABSTRACTS = safe_load_json(_CACHE_PATH, "abstracts") or {}


def load_papers(csv_path):
    papers = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            doi = row["article_doi"]
            if not doi:
                continue
            if doi not in papers:
                papers[doi] = {
                    "doi": doi,
                    "title": row["article_title"],
                    "authors": row["article_authors"],
                    "year": row["article_year"],
                    "datasets": [],
                }
            if row["dataset_doi"]:
                papers[doi]["datasets"].append({
                    "doi": row["dataset_doi"],
                    "title": row["dataset_title"],
                })
    return list(papers.values())


def build_index(papers):
    title_tokens = [p["title"].lower().split() for p in papers]
    title_bm25 = BM25Okapi(title_tokens)

    abstract_tokens = [
        _ABSTRACTS.get(p["doi"].lower(), {}).get("abstract", "").lower().split()
        or [""]
        for p in papers
    ]
    abstract_bm25 = BM25Okapi(abstract_tokens)

    return title_bm25, abstract_bm25


def search(query, papers, index, top_k=5):
    title_bm25, abstract_bm25 = index
    tokens = query.lower().split()

    title_scores = title_bm25.get_scores(tokens)
    abstract_scores = abstract_bm25.get_scores(tokens)

    use_abstracts = max(abstract_scores) > max(title_scores)
    scores = abstract_scores if use_abstracts else title_scores
    search_field = "abstract" if use_abstracts else "title"

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        paper = dict(papers[idx])
        apply_cache(paper, paper["doi"].lower())
        paper["matched_on"] = search_field
        results.append(paper)
    return results


def apply_cache(paper: dict, doi: str) -> None:
    """Overlay the cached abstract and OpenAlex metadata onto a paper dict.

    Adds abstract/abstract_source plus full authors (the CSV's `authors` field
    is truncated for non-ASCII surnames; OpenAlex is clean), journal, and
    citation_count. Falls back silently to the CSV fields when the DOI isn't
    cached or OpenAlex didn't know it.
    """
    cached = _ABSTRACTS.get(doi)
    if not cached:
        return
    if cached.get("abstract"):
        paper["abstract"] = cached["abstract"]
        paper["abstract_source"] = cached["source"]
    if cached.get("authors"):
        paper["authors"] = cached["authors"]
        paper["authors_source"] = "openalex"
    if cached.get("journal"):
        paper["journal"] = cached["journal"]
    if cached.get("citation_count") is not None:
        paper["citation_count"] = cached["citation_count"]