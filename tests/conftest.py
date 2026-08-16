"""Test fixtures shared across the suite.

Tests import the domain modules (search, graph, nomad_search, server, ...)
directly — never through app.py — so they exercise the reusable core and will
serve as the regression harness if the Streamlit UI is later replaced.

Running them requires the data caches to be built (see build.py); they assert
behavior against the real corpus, not fixtures.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
