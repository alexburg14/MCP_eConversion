"""Scrape cluster-level info from the e-conversion website into cluster_info.json.

Complements the paper/PI scrapers: this captures what the *cluster* is, not its
publications. Four sections, one page each (server-side rendered HTML, regex
parsing to match build_pis_cache.py):

  - about            : mission / what the cluster researches / partner institutions
  - research_areas   : RA 1..N with area coordinators and member PIs
  - governance       : executive board, cluster office, organization chart
  - news             : recent news headlines + links

The result backs the get_cluster_info(topic) MCP tool, so the agent can answer
"what are the research areas?", "who runs the cluster?", "which area is Sharp
in?", "what's new?" without these facts being hard-coded.

Usage:
    python src/scripts/build_cluster_info_cache.py
"""
import html as htmllib
import io
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
OUTPUT = DATA_DIR / "cache" / "cluster_info.json"
BASE = "https://www.e-conversion.de"
SLEEP = 0.4

PAGES = {
    "about": f"{BASE}/about-e-conversion/",
    "research_areas": f"{BASE}/research-areas/",
    "executive_board": f"{BASE}/executive-board/",
    "cluster_office": f"{BASE}/cluster-office/",
    "news": f"{BASE}/news/",
}

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 "
    "(mailto:audit@econversion.de)"
)
HEADERS = {"User-Agent": USER_AGENT}
# WordPress author + ISO-timestamp byline every page opens with, e.g.
# "Über uns Peter Sonntag 2025-11-25T13:39:39+01:00".
_BYLINE = re.compile(r"^.*?\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{2}:\d{2}\s*")


def fetch(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


def _main(html: str) -> str:
    m = re.search(r"<main[^>]*>(.*?)</main>", html, re.S)
    return m.group(1) if m else html


def _clean(seg: str, keep_breaks: bool = False) -> str:
    seg = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", seg, flags=re.S)
    if keep_breaks:
        seg = re.sub(r"(?i)<(?:br|/p|/div|/li|/h[1-6]|/tr)\s*/?>", "\n", seg)
    seg = re.sub(r"<[^>]+>", " ", seg)
    seg = htmllib.unescape(seg)
    if keep_breaks:
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in seg.split("\n")]
        return "\n".join(ln for ln in lines if ln)
    return re.sub(r"\s+", " ", seg).strip()


def _prose(html: str, limit: int = 2500) -> str:
    txt = _BYLINE.sub("", _clean(_main(html)))
    return txt[:limit].strip()


def parse_research_areas(html: str) -> list[dict]:
    txt = _BYLINE.sub("", _clean(_main(html), keep_breaks=True))
    areas, cur, expect_coord = [], None, False
    for line in txt.split("\n"):
        head = re.match(r"RA\s*(\d+)\s*:\s*(.+)", line)
        if head:
            cur = {"id": f"RA {head.group(1)}", "title": head.group(2).strip(),
                   "coordinators": [], "members": []}
            areas.append(cur)
            expect_coord = False
            continue
        if cur is None:
            continue
        if re.match(r"(?i)area coordinators", line):
            rest = re.sub(r"(?i)area coordinators\s*:?", "", line).strip()
            if rest:
                cur["coordinators"] = [c.strip() for c in rest.split(",") if c.strip()]
            else:
                expect_coord = True
            continue
        if expect_coord:
            cur["coordinators"] = [c.strip() for c in line.split(",") if c.strip()]
            expect_coord = False
            continue
        # member surnames are one per line; skip anything sentence-like
        if 1 < len(line) <= 40 and not re.search(r"[.:;!?]", line):
            cur["members"].append(line.strip())
    return [a for a in areas if a["members"] or a["coordinators"]]


def parse_news(html: str, limit: int = 15) -> list[dict]:
    skip = ("/category/", "/author/", "/tag/", "/page/", "/research-areas",
            "/members", "/publications", "/publikationen", "/about", "/news",
            "/events", "/newsletter", "/feed", "/wp-", "/executive-board",
            "/cluster-office", "/organization-chart", "/awards", "/logo-")
    out, seen = [], set()
    for href, label in re.findall(
        r'<a[^>]+href="(https://www\.e-conversion\.de/[a-z0-9\-]+/)"[^>]*>\s*([^<]{10,150})</a>', html
    ):
        if href in seen or any(s in href for s in skip):
            continue
        title = htmllib.unescape(label).strip()
        if not title or title.lower() in ("read more", "weiterlesen"):
            continue
        seen.add(href)
        out.append({"title": title, "url": href})
        if len(out) >= limit:
            break
    return out


def main() -> None:
    print("Scraping cluster info from", BASE)
    html = {name: fetch(url) for name, url in PAGES.items() for _ in [time.sleep(SLEEP)]}

    info = {
        "fetched_at": date.today().isoformat(),
        "sources": PAGES,
        "about": _prose(html["about"], 2500),
        "research_areas": parse_research_areas(html["research_areas"]),
        "governance": {
            "executive_board": _prose(html["executive_board"], 1800),
            "cluster_office": _prose(html["cluster_office"], 1800),
        },
        "news": parse_news(html["news"]),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    ra = info["research_areas"]
    print(f"\nwrote {OUTPUT}")
    print(f"  about        : {len(info['about'])} chars")
    print(f"  research areas: {len(ra)} " + "; ".join(
        f"{a['id']} ({len(a['members'])} members)" for a in ra))
    print(f"  governance   : " + ", ".join(
        f"{k} {len(v)}c" for k, v in info["governance"].items()))
    print(f"  news items   : {len(info['news'])}")


if __name__ == "__main__":
    main()
