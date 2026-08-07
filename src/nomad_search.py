"""Live search over public NOMAD materials-science entries.

Unlike every other tool here, this one hits a remote API per call rather than a
local cache — NOMAD's corpus is millions of entries and changes continuously, so
snapshotting it is not practical.

No credentials: NOMAD's OpenAPI spec declares OAuth2 on every route, but auth is
optional and only widens *scope*. Unauthenticated requests resolve to
`owner="public"` (published, no embargo), which is all we want. Sending a token
we don't need would only add a failure mode — an expired one 401s queries that
work fine anonymously.

Results are NOT joinable to the cluster's papers: NOMAD's `references` field is
an exact-match URL string chosen by whoever deposited the entry, so neither a
bare DOI nor a doi.org URL matches it. `authors.name` is the only reliable link
back to a PI.
"""
import requests

_API = "https://nomad-lab.eu/prod/v1/api/v1/entries/query"
_GUI = "https://nomad-lab.eu/prod/v1/gui/entry/id/{}"
_TIMEOUT = 20

# Returned per hit. crystal_system is null for molecules/clusters and for entries
# whose parser never resolved symmetry — that absence is informative, not a gap.
_FIELDS = [
    "entry_id",
    "results.material.chemical_formula_reduced",
    "results.material.symmetry.crystal_system",
    "authors.name",
    "references",
    "upload_create_time",
]


def search_nomad(
    elements: str = "",
    formula: str = "",
    author: str = "",
    text: str = "",
    limit: int = 5,
) -> dict:
    """Query public NOMAD. Filters combine with AND; at least one is required.

    `elements` is comma-separated ("Ti,O") and matches entries containing ALL of them.
    `formula` matches NOMAD's reduced form (alphabetical, e.g. SrTiO3 -> "O3SrTi").
    """
    query: dict = {}
    element_list = [e.strip() for e in elements.split(",") if e.strip()]
    if element_list:
        query["results.material.elements"] = {"all": element_list}
    if formula.strip():
        query["results.material.chemical_formula_reduced"] = formula.strip()
    if author.strip():
        query["authors.name"] = author.strip()
    if text.strip():
        query["text_search_contents"] = text.strip()

    if not query:
        return {"error": "Provide at least one filter: elements, formula, author, or text."}

    # Default ordering is entry_id ascending, which always returns the same
    # lexicographic corner of the index regardless of the query. Score-rank text
    # queries; fall back to newest-first when there is no text to score.
    order_by = "_score" if text.strip() else "upload_create_time"

    payload = {
        "query": query,
        "pagination": {"page_size": limit, "order_by": order_by, "order": "desc"},
        "required": {"include": _FIELDS},
    }

    try:
        response = requests.post(_API, json=payload, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        return {"error": f"Could not reach NOMAD: {exc}"}

    if response.status_code == 422:
        return {"error": "NOMAD rejected the query.", "detail": response.json().get("detail")}
    if response.status_code != 200:
        return {"error": f"NOMAD returned HTTP {response.status_code}: {response.text[:200]}"}

    body = response.json()
    entries = []
    for entry in body.get("data", []):
        material = (entry.get("results") or {}).get("material") or {}
        entries.append({
            "entry_id": entry.get("entry_id"),
            "formula": material.get("chemical_formula_reduced"),
            "crystal_system": (material.get("symmetry") or {}).get("crystal_system"),
            "authors": [a.get("name") for a in entry.get("authors", [])],
            "references": entry.get("references", []),
            "upload_create_time": entry.get("upload_create_time"),
            "url": _GUI.format(entry.get("entry_id")),
        })

    return {
        "filters": {"elements": elements, "formula": formula, "author": author, "text": text},
        "ranked_by": order_by,
        "total_matches": body["pagination"]["total"],
        "returned": len(entries),
        "entries": entries,
    }
