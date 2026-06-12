"""Build embeddings_cache.npz — dense vectors for semantic paper search.

Encodes "{title}. {abstract}" for every paper in abstracts_cache.json with the
BGE-small retrieval model (384-d), L2-normalises so cosine == dot product, and
writes a single .npz with parallel `dois` and `vectors` arrays.

Re-run only when abstracts_cache.json changes. The corpus (~950 docs) fits in
memory; no vector DB needed.
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
ABSTRACTS = DATA_DIR / "abstracts_cache.json"
CSV_PATH = DATA_DIR / "data_publication_dois.csv"
OUTPUT = DATA_DIR / "embeddings_cache.npz"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 32


def load_titles_by_doi() -> dict:
    titles = {}
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            doi = row["article_doi"].lower() if row["article_doi"] else None
            if doi and doi not in titles:
                titles[doi] = row["article_title"] or ""
    return titles


def main():
    if not ABSTRACTS.exists():
        raise SystemExit(f"Missing {ABSTRACTS.name}. Run build_abstracts_cache.py first.")
    with open(ABSTRACTS, encoding="utf-8") as f:
        abstracts = json.load(f)

    titles = load_titles_by_doi()

    dois = sorted(abstracts.keys())
    texts = [f"{titles.get(d, '').strip()}. {abstracts[d]['abstract'].strip()}".strip(". ") for d in dois]

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Encoding {len(texts)} papers...")
    t0 = time.time()
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"Encoded in {time.time() - t0:.1f}s. Shape: {vectors.shape}")

    np.savez(OUTPUT, dois=np.array(dois, dtype=object), vectors=vectors, model=MODEL_NAME)
    print(f"Wrote {OUTPUT.name}")


if __name__ == "__main__":
    main()
