"""filter_candidates.py — keyword-based relevance filtering for video candidate CSV.

Reads `jab_video_candidates.csv` (or any candidate CSV), evaluates each row's
title and tags against allow/reject keyword lists, then writes back the `keep`
column with 'yes' or 'no'.  A summary is printed to stdout.

Usage (defaults work out of the box):
    python filter_candidates.py
    python filter_candidates.py --input-csv reference_poses/jab_video_candidates.csv
    python filter_candidates.py --dry-run          # print decisions without writing
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# ── Keyword lists ──────────────────────────────────────────────────────────────
# Words / phrases that strongly suggest a genuine boxing/martial-arts technique
# video.  Checked against: title + tags (case-insensitive).
KEEP_PATTERNS: list[str] = [
    r"\bjab\b",
    r"\bpunch(ing|er)?\b",
    r"\bbox(ing|er)?\b",
    r"muay[ -]?thai",
    r"kickbox(ing|er)?",
    r"\bmma\b",
    r"martial[ -]?art",
    r"\bstrike\b",
    r"\bstrik(ing|er)?\b",
    r"left[ -]?hook",
    r"right[ -]?hook",
    r"\bhook\b",
    r"\buppercut\b",
    r"\bcross\b.*punch|punch.*\bcross\b",
    r"one[-\s]two",
    r"\bcombo\b",
    r"slow[ -]?motion.*punch|punch.*slow[ -]?motion",
    r"how to (throw|land|throw a|punch)",
    r"boxing (technique|form|lesson|tutorial|training|drill|fundamentals)",
    r"(technique|tutorial|form|mechanic|drill).*punch",
    r"southpaw",
    r"orthodox (stance|style|boxing)",
    r"fighting stance",
    r"\bknockout\b",
    r"\bko\b",
    r"sparring",
    r"boxing workout",
    r"boxing (for|class|camp)",
]

# Words / phrases that strongly indicate a NON-technique video even if a keep
# word crept in (e.g. the query "jab form slow motion" returned caterpillar
# time-lapses because YouTube also matched unrelated words).
REJECT_PATTERNS: list[str] = [
    r"time[ -]?lapse",
    r"timelapse",
    r"\bcaterpillar\b",
    r"\bbutterfly\b",
    r"\bpeanut\b",
    r"\bplant\b",
    r"growing (plant|seed|flower|vegetable)",
    r"\bcricket\b",
    r"\bbasketball\b",
    r"\bnba\b",
    r"\bnfl\b",
    r"chess",
    r"rubik",
    r"long[ -]?jump",
    r"high[ -]?jump",
    r"triple[ -]?jump",
    r"\bgoal\b.*soccer|soccer.*\bgoal\b",
    r"\bsoccer\b",
    r"\bfootball\b(?!.*box)",   # football but not "football boxing"
    r"\bbaseball\b",
    r"\bhockey\b",
    r"\btennis\b",
    r"ping[ -]?pong",
    r"\bvolleyball\b",
    r"\bswimming\b",
    r"\bgolf\b",
    r"\bcycling\b",
    r"lamelo",
    r"kohli|virat",
    r"minecraft",
    r"\bgaming\b",
    r"\bfortnite\b",
    r"dance|dancing",
    r"cooking|recipe|food",
    r"makeup|beauty",
    r"prank",
    r"magic trick",
    r"rubix|magic cube",
    r"water.*fast(er|est)|fast(er|est).*water",
]

_KEEP_RE = [re.compile(p, re.IGNORECASE) for p in KEEP_PATTERNS]
_REJECT_RE = [re.compile(p, re.IGNORECASE) for p in REJECT_PATTERNS]


def _keep_text(row: dict) -> str:
    """Text to check for KEEP patterns — title and tags only, NOT query_hit.

    query_hit contains the search query that surfaced the video (e.g. "jab form
    slow motion"), so it would trivially match \bjab\b for every row and cause
    false-positives for completely unrelated videos.
    """
    parts = [
        row.get("title", ""),
        row.get("tags", ""),
    ]
    return " ".join(p for p in parts if p)


def _reject_text(row: dict) -> str:
    """Text to check for REJECT patterns — title, tags, and notes."""
    parts = [
        row.get("title", ""),
        row.get("tags", ""),
        row.get("notes", ""),
    ]
    return " ".join(p for p in parts if p)


def classify_row(row: dict) -> tuple[str, str]:
    """Return ('yes'|'no', reason_string)."""
    reject_text = _reject_text(row)
    keep_text = _keep_text(row)

    # Hard reject first — even if a keep keyword accidentally appears
    for r in _REJECT_RE:
        if r.search(reject_text):
            return "no", f"matched reject pattern: {r.pattern}"

    # Now look for positive signal in title/tags only
    for r in _KEEP_RE:
        if r.search(keep_text):
            return "yes", f"matched keep pattern: {r.pattern}"

    return "no", "no keep keyword found in title/tags"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Keyword-filter a video candidate CSV and set the 'keep' column."
    )
    p.add_argument(
        "--input-csv",
        default="reference_poses/jab_video_candidates.csv",
        help="Candidate CSV to filter.",
    )
    p.add_argument(
        "--output-csv",
        default="",
        help="Where to write the filtered CSV. Defaults to overwriting --input-csv.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print decisions without writing any file.",
    )
    p.add_argument(
        "--show-kept",
        action="store_true",
        help="Print every row that will be kept.",
    )
    p.add_argument(
        "--show-rejected",
        action="store_true",
        help="Print every row that will be rejected.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv) if args.output_csv else input_path

    if not input_path.exists():
        raise SystemExit(f"ERROR: input CSV not found: {input_path}")

    rows: list[dict] = []
    with input_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(row)

    kept, rejected = 0, 0
    for row in rows:
        decision, reason = classify_row(row)
        row["keep"] = decision
        if decision == "yes":
            kept += 1
            if args.show_kept:
                print(f"  KEEP  | {row.get('title', '')[:70]}")
                print(f"        | reason: {reason}")
        else:
            rejected += 1
            if args.show_rejected:
                print(f"  DROP  | {row.get('title', '')[:70]}")
                print(f"        | reason: {reason}")

    print(f"\nFilter results: {kept} kept, {rejected} rejected out of {len(rows)} total rows.")

    if args.dry_run:
        print("Dry-run: no file written.")
        return

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
