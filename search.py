import csv
import requests
from rank_bm25 import BM25Okapi


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
    tokenized = [p["title"].lower().split() for p in papers]
    return BM25Okapi(tokenized)


def _reconstruct_abstract(inverted_index):
    if not inverted_index:
        return None
    positions = {}
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions[pos] = word
    return " ".join(positions[i] for i in sorted(positions))


def fetch_openalex(doi, email="alexburg777@gmail.com"):
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    try:
        r = requests.get(url, headers={"User-Agent": f"mailto:{email}"}, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        return {
            "abstract": _reconstruct_abstract(data.get("abstract_inverted_index")),
            "open_access": data.get("open_access", {}).get("is_oa", False),
            "openalex_url": data.get("id"),
        }
    except Exception:
        return None


def search(query, papers, bm25, top_k=5):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        paper = dict(papers[idx])
        extra = fetch_openalex(paper["doi"])
        if extra:
            paper.update(extra)
        results.append(paper)
    return results
