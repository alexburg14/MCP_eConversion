"""Extract the 'Summary of the Proposal' section from the e-conversion 2.0 proposal PDF.

Writes data/proposal_summary.md (~6 KB, ~1.5K tokens) for use as system-prompt context
in the chat interface. Re-run only if the proposal PDF changes.

Usage:
    python src/extract_proposal_summary.py
"""
import re
import sys
from pathlib import Path

import pymupdf

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PROPOSAL_PDF = _DATA_DIR / "sources" / "EXC_2089_e-conversion_A_Proposal_R.pdf"
OUTPUT_MD = _DATA_DIR / "cache" / "proposal_summary.md"


def _extract_full_text(pdf_path: Path) -> str:
    doc = pymupdf.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


def _slice_section_2(text: str) -> str:
    """Return the body of Section 2 'Summary of the Proposal' (skipping the TOC entry)."""
    starts = [m.start() for m in re.finditer(r"2\s+\n?Summary of the Proposal", text)]
    if len(starts) < 2:
        raise RuntimeError("Could not locate Section 2 in the proposal (TOC + body expected).")
    body_start = starts[1]  # First hit is the TOC, second is the real section
    end_hits = [m.start() for m in re.finditer(r"3\s+\n?Objectives of the Cluster", text[body_start:])]
    if not end_hits:
        raise RuntimeError("Could not locate Section 3 boundary after Section 2.")
    return text[body_start : body_start + end_hits[0]].strip()


def _clean(text: str) -> str:
    # Strip DFG-form page footers
    text = re.sub(r"DFG form exstra130[^\n]*\n+page \d+ of \d+\n*", "", text)
    text = re.sub(r"\n+\s*page \d+ of \d+\s*\n+", "\n", text)
    # Join words split across line wraps: "con-\nversion" -> "conversion"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def main() -> None:
    if not PROPOSAL_PDF.exists():
        print(f"ERROR: {PROPOSAL_PDF} not found.", file=sys.stderr)
        sys.exit(1)

    full = _extract_full_text(PROPOSAL_PDF)
    summary = _clean(_slice_section_2(full))
    OUTPUT_MD.write_text(summary, encoding="utf-8")
    print(f"Wrote {OUTPUT_MD} ({len(summary)} chars, ~{len(summary) // 4} tokens)")


if __name__ == "__main__":
    main()
