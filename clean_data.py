"""One-shot CSV cleaner: decode LaTeX accent escapes in author/title fields.

Handles patterns like \\'{i}, \\"{u}, \\^{e} including truncated variants
where the closing brace was cut off by the original export script.
"""
import csv
import re
from pathlib import Path

ACCENTS = {
    "'": {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú", "y": "ý", "c": "ć", "n": "ń", "s": "ś", "z": "ź",
          "A": "Á", "E": "É", "I": "Í", "O": "Ó", "U": "Ú", "Y": "Ý"},
    '"': {"a": "ä", "e": "ë", "i": "ï", "o": "ö", "u": "ü", "y": "ÿ",
          "A": "Ä", "E": "Ë", "I": "Ï", "O": "Ö", "U": "Ü"},
    "^": {"a": "â", "e": "ê", "i": "î", "o": "ô", "u": "û",
          "A": "Â", "E": "Ê", "I": "Î", "O": "Ô", "U": "Û"},
    "`": {"a": "à", "e": "è", "i": "ì", "o": "ò", "u": "ù",
          "A": "À", "E": "È", "I": "Ì", "O": "Ò", "U": "Ù"},
    "~": {"a": "ã", "n": "ñ", "o": "õ", "A": "Ã", "N": "Ñ", "O": "Õ"},
    "c": {"c": "ç", "C": "Ç"},
    "v": {"c": "č", "s": "š", "z": "ž", "C": "Č", "S": "Š", "Z": "Ž"},
}

LATEX_PATTERN = re.compile(r"\\(['\"\^`~cv])\{([a-zA-Z])\}?")


def _replace(match):
    accent, letter = match.group(1), match.group(2)
    return ACCENTS.get(accent, {}).get(letter, letter)


def clean(text):
    if not text:
        return text
    cleaned = LATEX_PATTERN.sub(_replace, text)
    # Strip stray unbalanced braces left behind
    cleaned = cleaned.replace("\\\\", "").replace("{", "").replace("}", "")
    return cleaned


def main():
    src = Path(__file__).parent / "data_publication_dois.csv"
    rows_changed = 0
    total_rows = 0

    with open(src, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    for row in rows:
        total_rows += 1
        before = (row["article_title"], row["article_authors"])
        row["article_title"] = clean(row["article_title"])
        row["article_authors"] = clean(row["article_authors"])
        if (row["article_title"], row["article_authors"]) != before:
            rows_changed += 1

    with open(src, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Cleaned {rows_changed} of {total_rows} rows.")


if __name__ == "__main__":
    main()
