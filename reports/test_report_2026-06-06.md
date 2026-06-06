# PI Tool Hardening — Stress Test Report — 2026-06-06

Follow-up to `test_report_2026-06-05.md`. A live stress battery against `search_pis`
and `get_pi` exposed six bugs (3 orange, 2 yellow, 1 red). Items #1–#4 were
implemented in this pass; #5 and #6 are noted for later.

All fixes live in `src/server.py`; no schema changes.

---

## Implemented

### 🔴 #1 — Accent-insensitive matching

**Before:** `get_pi("Cortes")` → error. `search_pis("muller")` → no matches.
**After:** ASCII spellings find their accented counterparts.

**Fix:** New `_fold(s)` helper (NFKD decompose + strip combining marks + lower).
Applied to query, PI text fields, and last-name comparisons.

| Query | Before | After |
|---|---|---|
| `get_pi("Cortes")` | error | Prof. Dr. Emiliano Cortés ✅ |
| `get_pi("muller")` | error | disambiguation (3 candidates: Koblmüller, Müller-Buschbaum, Müller-Caspary) ✅ |
| `get_pi("Roldan")` | error | Prof. Dr. Beatriz Roldán Cuenya ✅ |
| `search_pis("muller")` | 0 results | Müller-Buschbaum, Müller-Caspary ✅ |

---

### 🟠 #2 — Empty/blank `get_pi` returns first PI

**Before:** `get_pi("")` → Dr. Erkan Aydin (alphabetically first), because `"" in pi["name"]` is always True.
**After:** Empty/whitespace input returns a clean `"Name must not be empty."` error.

`search_pis` was already guarded; this just brings `get_pi` into line.

---

### 🟠 #3 — Substring + stopword pollution

**Before:** `search_pis("Müller")` returned Koblmüller (substring of "koblmüller").
`search_pis("the of and a in for")` returned 5 unrelated PIs (1-2 char tokens matched as substrings; "the"/"and"/"for" matched real text).

**After:**
- Word-token matching: `Müller-Buschbaum` tokenizes to `{müller, buschbaum}`; `Koblmüller` stays `{koblmüller}` — no spurious match.
- Query tokenizer drops tokens shorter than 3 chars and a tiny English stopword set (`the, and, for, with, from, this, that, ...` — ~40 common fillers).

| Query | Before | After |
|---|---|---|
| `search_pis("Müller")` | included Koblmüller | only Müller-Buschbaum, Müller-Caspary ✅ |
| `search_pis("the of and a in for")` | 5 PIs | `no results, message: "all tokens too short or stopwords"` ✅ |
| `search_pis("energy conversion")` | (worked) | still works ✅ — domain words not in stopword set |
| `search_pis("machine learning")` | (worked) | still works ✅ |

---

### 🟠 #4 — Duplicate last name "Stein"

**Before:** `get_pi("Stein")` silently returned Christopher Stein; Helge Stein was unreachable by last name alone.
**After:** Last-name collisions return a disambiguation payload listing all candidates with their groups, prompting the caller to specify a full name.

```json
{
  "error": "Multiple PIs share the last name 'Stein'. Specify the full name.",
  "matches": [
    {"name": "Prof. Dr. Christopher Stein", "group": "Associate Professorship of Theoretical Chemistry", ...},
    {"name": "Prof. Dr. Helge Stein",       "group": "Chair of Digital Catalysis", ...}
  ]
}
```

`get_pi("Helge Stein")` and `get_pi("Christopher Stein")` resolve unambiguously. Same disambiguation logic also fires on the substring step if a partial query matches multiple PIs.

---

## Not yet done (deferred)

### 🟡 #5 — No relevance tie-break in `search_pis`

`search_pis("photovoltaics")` still ranks Aydin (0 publications) above Müller-Buschbaum (458 publications) because both score 1 token-match and ties fall to file order. **Proposed fix:** secondary sort by `publication_count`. Held back from this pass to keep the change surgical — would shift ranking globally and deserves its own evaluation.

### 🟡 #6 — `get_pi` returns a person for topic queries

`get_pi("catalysis")` still returns Bandarenka via the keyword-scoring fallback. The tool contract says "by name" — arguably topic queries should error out and direct the caller to `search_pis`. Held back because this is a behavioral contract change, not a bug fix.

---

## Regression checks (all green)

- `get_pi("Rinke")` → Prof. Dr. Patrick Rinke ✅
- `get_pi("cuenya")` → Prof. Dr. Beatriz Roldán Cuenya (partial-name substring) ✅
- `get_pi("qwerty zxcvb")` → clean "No PI found" ✅
- `search_pis("solar cells photovoltaics")` → 5 relevant PIs ✅
- `search_pis("")` → clean "must not be empty" ✅
