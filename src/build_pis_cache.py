"""Build pis_cache.json by scraping the eConversion members directory.

Two stages:
  1. Fetch https://www.e-conversion.de/members/ and parse one card per PI
     (name parts, group, department, institution, group website, image, smid).
  2. For each PI, fetch the single-staff-page and extract:
     - Academic Research Focus (bullet list)
     - Fields of Application (comma-separated)
     - publication DOIs found in the Publications section

Pages are server-side rendered HTML — no JS, no API. Parsing is regex-based to
stay dependency-free, matching the style of the other build_*_cache.py scripts.
"""
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

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT = DATA_DIR / "pis_cache.json"
LISTING_URL = "https://www.e-conversion.de/members/"
PROFILE_URL_TMPL = "https://www.e-conversion.de/single-staff-page/?smid={smid}"
FORCE = "--force" in sys.argv
SLEEP = 0.4

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 "
    "(mailto:audit@econversion.de)"
)
HEADERS = {"User-Agent": USER_AGENT}


def fetch(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


def _text(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()


def parse_listing(html: str) -> list[dict]:
    """Each PI card is anchored by an <h3 class="...MP-F1"> containing three spans
    (title prefix, first name, last name) wrapped in a link to the profile."""
    starts = [m.start() for m in re.finditer(r'<h3[^>]*MP-F1[^>]*>', html)]
    starts.append(len(html))
    cards = []
    for i in range(len(starts) - 1):
        seg = html[starts[i]:starts[i + 1]]

        m = re.search(
            r'<a href="(?P<profile>[^"]*smid=(?P<smid>\d+))"[^>]*>'
            r'\s*<span class="abcfslSpanMP1">(?P<title>[^<]*)</span>'
            r'\s*<span class="abcfslSpanMP2">(?P<first>[^<]*)</span>'
            r'\s*<span class="abcfslSpanMP3">(?P<last>[^<]*)</span>',
            seg,
        )
        if not m:
            continue

        def field(cls):
            mm = re.search(rf'<div class="{cls}">(.*?)</div>', seg, re.S)
            return _text(mm.group(1)) if mm else ""

        website = ""
        mm = re.search(r'<div class="TH-F6">\s*<a href="([^"]+)"', seg)
        if mm:
            website = mm.group(1)

        # Image lives in the card directly preceding the heading; look backwards.
        img = ""
        prev = html[max(0, starts[i] - 1500):starts[i]]
        mm = re.search(rf'<img[^>]*src="([^"]+)"[^>]*alt="[^"]*"[^>]*itemprop="image"', prev)
        if mm:
            img = mm.group(1)

        title = m.group("title").strip()
        first = m.group("first").strip()
        last = m.group("last").strip()
        cards.append({
            "smid": m.group("smid"),
            "name": f"{title} {first} {last}".strip(),
            "title": title,
            "first_name": first,
            "last_name": last,
            "group": field("T-F2"),
            "department": field("CBO-F3"),
            "institution": field("CBO-F4"),
            "website": website,
            "profile_url": f"https://www.e-conversion.de/single-staff-page/?smid={m.group('smid')}",
            "image_url": img,
        })
    return cards


def parse_profile(html: str) -> dict:
    research_focus: list[str] = []
    m = re.search(r'Academic Research Focus</div>\s*<div class="abcfslMT5">(.*?)</div>', html, re.S)
    if m:
        for line in re.split(r'<br\s*/?>', m.group(1)):
            t = _text(line)
            t = re.sub(r'^[-–—•\s]+', '', t).strip()
            if t:
                research_focus.append(t)

    application_fields: list[str] = []
    m = re.search(
        r'Fields of Application</div>\s*<div class="[^"]*STFFCAT[^"]*">(.*?)</div>',
        html, re.S,
    )
    if m:
        application_fields = [s.strip() for s in _text(m.group(1)).split(",") if s.strip()]

    # Restrict DOI harvesting to the Publications block so we don't catch
    # anything from the page chrome (currently none, but future-proofing).
    pubs_idx = html.find('>Publications<')
    pub_html = html[pubs_idx:] if pubs_idx > 0 else html
    raw = re.findall(r'10\.\d{4,9}/[^\s"<>]+', pub_html)
    dois = sorted({re.sub(r'[.,;)\]]+$', '', d).lower() for d in raw})

    return {
        "research_focus": research_focus,
        "application_fields": application_fields,
        "publication_dois": dois,
    }


def main() -> None:
    if OUTPUT.exists() and not FORCE:
        existing = json.loads(OUTPUT.read_text(encoding="utf-8"))
        cached_smids = {pi["smid"] for pi in existing}
        print(f"existing cache: {len(existing)} PIs (pass --force to re-fetch)")
    else:
        existing = []
        cached_smids = set()

    print(f"fetching listing: {LISTING_URL}")
    listing_html = fetch(LISTING_URL)
    cards = parse_listing(listing_html)
    print(f"parsed {len(cards)} PIs from listing")

    by_smid = {pi["smid"]: pi for pi in existing}
    for i, card in enumerate(cards, 1):
        smid = card["smid"]
        if smid in cached_smids and not FORCE:
            by_smid[smid].update({k: v for k, v in card.items() if v})
            continue

        url = PROFILE_URL_TMPL.format(smid=smid)
        print(f"[{i:>2}/{len(cards)}] {card['name']} (smid={smid})")
        try:
            profile_html = fetch(url)
        except requests.RequestException as e:
            print(f"  ! fetch failed: {e}")
            by_smid[smid] = {**card, "research_focus": [], "application_fields": [],
                             "publication_dois": [], "fetched_at": None, "error": str(e)}
            continue

        card.update(parse_profile(profile_html))
        card["fetched_at"] = date.today().isoformat()
        by_smid[smid] = card
        time.sleep(SLEEP)

    pis = sorted(by_smid.values(), key=lambda p: p.get("last_name", ""))
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(pis, indent=2, ensure_ascii=False), encoding="utf-8")

    with_dois = sum(1 for p in pis if p.get("publication_dois"))
    total_dois = sum(len(p.get("publication_dois", [])) for p in pis)
    print(f"\nwrote {len(pis)} PIs to {OUTPUT.name}")
    print(f"  {with_dois} have publication DOIs ({total_dois} total, "
          f"{len({d for p in pis for d in p.get('publication_dois', [])})} unique)")


if __name__ == "__main__":
    main()
