"""Precompute the UMAP projection into data/cache/corpus_projection.npz.

UMAP costs ~30 s cold (numba JIT included) and Streamlit drops its caches on every
deploy, so without this file the first visitor after each deploy pays it again.

Run inside the container:  python src/scripts/build_corpus_map_cache.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import corpus_map  # noqa: E402

started = time.perf_counter()
coords = corpus_map._project()
print("projection %s in %.1f s -> %s" % (coords.shape, time.perf_counter() - started,
                                         corpus_map._PROJ_PATH))
