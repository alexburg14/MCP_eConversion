"""Probe one missing-DOI per publisher to diagnose why full text is unreachable.

For every DOI prefix in the missing set (papers with no fulltext_cache entry),
resolves one example DOI via doi.org and reports status, WAF evidence
(server / cf-mitigated headers, challenge-page title), and body size —
distinguishing Cloudflare bot challenges from paywalls from open doors.

Usage:
    python src/scripts/probe_publishers.py [--proxy http://host:port] [--sleep N]

--proxy routes the probes through a forward proxy (e.g. a VPN'd host or SSH
tunnel) so access can be compared across networks. The exit IP is printed
first so every probe run records which network it measured.
"""
import csv
import json
import re
import sys
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CSV_PATH = DATA_DIR / "data_publication_dois.csv"
FULLTEXT = DATA_DIR / "fulltext_cache.json"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

PREFIX_NAMES = {
    "10.1021": "ACS", "10.1039": "RSC", "10.1002": "Wiley", "10.1063": "AIP",
    "10.1038": "Nature", "10.1103": "APS", "10.1088": "IOP", "10.1016": "Elsevier",
    "10.26434": "ChemRxiv", "10.1149": "ECS", "10.1364": "Optica", "10.1007": "Springer",
    "10.1109": "IEEE", "10.29026": "OEA", "10.1055": "Thieme", "10.1107": "IUCr",
    "10.1117": "SPIE", "10.1017": "CUP", "10.1080": "T&F", "10.1093": "OUP",
    "10.1515": "DeGruyter", "10.1557": "MRS", "10.3390": "MDPI",
    "10.34133": "Science-Part", "10.48550": "arXiv",
}


def classify(r: requests.Response) -> str:
    if r.headers.get("cf-mitigated") == "challenge":
        return "CLOUDFLARE-CHALLENGE"
    if r.status_code == 403:
        return "blocked-403"
    if r.status_code == 202 and len(r.content) < 5000:
        return "bot-managed-202"
    if r.status_code == 200:
        return "open-200"
    return f"other-{r.status_code}"


def main():
    proxy = None
    if "--proxy" in sys.argv:
        try:
            proxy = sys.argv[sys.argv.index("--proxy") + 1]
        except IndexError:
            raise SystemExit("Usage: --proxy <url>")
    sleep = 6.0
    if "--sleep" in sys.argv:
        sleep = float(sys.argv[sys.argv.index("--sleep") + 1])

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}

    exit_ip = session.get("https://api.ipify.org", timeout=15).text
    print(f"exit IP: {exit_ip}   proxy: {proxy or '(none, direct)'}\n")

    with open(FULLTEXT, encoding="utf-8") as f:
        cached = set(json.load(f))
    with open(CSV_PATH, encoding="utf-8") as f:
        all_dois = {r["article_doi"].lower() for r in csv.DictReader(f) if r["article_doi"]}
    missing = sorted(all_dois - cached)

    targets = {}  # prefix -> (example_doi, count)
    for d in missing:
        p = d.split("/")[0]
        if p in targets:
            targets[p] = (targets[p][0], targets[p][1] + 1)
        else:
            targets[p] = (d, 1)

    print(f"{'publisher':14s} {'missing':>7s}  {'verdict':22s} {'status':>6s} {'bytes':>8s}  server")
    for prefix, (doi, count) in sorted(targets.items(), key=lambda kv: -kv[1][1]):
        name = PREFIX_NAMES.get(prefix, prefix)
        try:
            r = session.get(f"https://doi.org/{doi}", timeout=30, allow_redirects=True)
            verdict = classify(r)
            server = r.headers.get("server", "")
            print(f"{name:14s} {count:7d}  {verdict:22s} {r.status_code:6d} {len(r.content):8d}  {server}", flush=True)
        except Exception as exc:
            print(f"{name:14s} {count:7d}  ERROR-{type(exc).__name__}", flush=True)
        time.sleep(sleep)


if __name__ == "__main__":
    main()
