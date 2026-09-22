"""Warm everything expensive after a deploy so the first visitor does not wait.

Run inside the container:  python src/scripts/warmup.py

Loads, in this order: the caches + BM25 index (server import), the full-text cache
(background thread), the embedding model and the UMAP projection. Takes ~30-60 s on
a cold process and makes every later session cheap.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

overall = time.perf_counter()

started = time.perf_counter()
import server  # noqa: E402  (caches, BM25 index, tool registry)

print("caches_s=%.1f" % (time.perf_counter() - started))

import corpus_map  # noqa: E402
import semantic_search  # noqa: E402

started = time.perf_counter()
if server._FULLTEXTS_READY.wait(timeout=300):
    print("fulltexts_s=%.1f (%d entries)" % (time.perf_counter() - started, len(server._FULLTEXTS)))
else:
    print("fulltexts_s=timeout")

started = time.perf_counter()
semantic_search._load()
print("embedding_model_s=%.1f" % (time.perf_counter() - started))

started = time.perf_counter()
corpus_map._project()
print("umap_projection_s=%.1f" % (time.perf_counter() - started))

print("warm_total_s=%.1f" % (time.perf_counter() - overall))
