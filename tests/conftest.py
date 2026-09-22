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

from pytest import fixture  # noqa: E402  (imported here so the block below stays self-contained)

import mcp_clients  # noqa: E402  (sys.path is set above)


@fixture(autouse=True)
def _clear_remote_tools_cache():
    """The remote tool schema cache is process-global (it spans reruns by design);
    clear it so one test cannot answer for the next."""
    mcp_clients._TOOLS_CACHE.clear()
    yield
    mcp_clients._TOOLS_CACHE.clear()
