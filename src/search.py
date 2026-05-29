import csv
import json
from pathlib import Path
from rank_bm25 import BM25Okapi

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_CACHE_PATH = _DATA_DIR / "abstracts_cache.json"


def _load_abstracts_cache():
    if _CACHE_PATH.exists():
        with open(_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


_ABSTRACTS = _load_abstracts_cache()


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
        cached = _ABSTRACTS.get(paper["doi"].lower())
        if cached:
            paper["abstract"] = cached["abstract"]
            paper["abstract_source"] = cached["source"]
        paper["matched_on"] = search_field
        results.append(paper)
    return results