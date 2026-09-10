"""SQLite-backed candidate catalog for scouting, review, and plan export.

CSV remains an interchange format for the existing YouTube scout and batch
collector; the catalog is the persistent source of truth between those steps.

Examples:
    python scripts/candidate_catalog.py import-scout --csv reference_poses/scout_candidates_golden_seeds.csv
    python scripts/candidate_catalog.py list --technique jab --angle front
    python scripts/candidate_catalog.py review --technique jab --angle front --url URL --status approved
    python scripts/candidate_catalog.py export-plan --output-csv reference_poses/catalog_capture_plan.csv
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from scout_utils import create_csv_template_row, normalize_technique_key

DEFAULT_DB = "reference_poses/candidate_catalog.sqlite"
PLAN_FIELDNAMES = list(create_csv_template_row("jab", "front", ["https://example.com"]).keys())
VALID_STATUSES = ("pending", "approved", "rejected")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY,
            source_url TEXT NOT NULL,
            technique TEXT NOT NULL,
            angle TEXT NOT NULL,
            video_id TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '',
            channel_title TEXT NOT NULL DEFAULT '',
            channel_id TEXT NOT NULL DEFAULT '',
            published_at TEXT NOT NULL DEFAULT '',
            duration_seconds INTEGER,
            duration_iso8601 TEXT NOT NULL DEFAULT '',
            definition TEXT NOT NULL DEFAULT '',
            caption TEXT NOT NULL DEFAULT '',
            view_count INTEGER,
            like_count INTEGER,
            comment_count INTEGER,
            query_hit TEXT NOT NULL DEFAULT '',
            tags TEXT NOT NULL DEFAULT '',
            pose_score REAL,
            motion REAL,
            closure REAL,
            review_status TEXT NOT NULL DEFAULT 'pending'
                CHECK(review_status IN ('pending', 'approved', 'rejected')),
            reviewer_notes TEXT NOT NULL DEFAULT '',
            discovered_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(source_url, technique, angle)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS candidates_group_idx ON candidates(technique, angle, review_status)"
    )
    return connection


def int_or_none(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def float_or_none(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def import_scout_csv(connection: sqlite3.Connection, path: Path) -> int:
    imported = 0
    timestamp = now()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source_url = (row.get("url") or row.get("source_url") or "").strip()
            technique = normalize_technique_key(row.get("technique", ""))
            angle = normalize_technique_key(row.get("angle", ""))
            if not source_url or not technique or not angle:
                continue
            connection.execute(
                """
                INSERT INTO candidates (
                    source_url, technique, angle, video_id, title, channel_title,
                    channel_id, published_at, duration_seconds, duration_iso8601,
                    definition, caption, view_count, like_count, comment_count,
                    query_hit, tags, discovered_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_url, technique, angle) DO UPDATE SET
                    video_id=excluded.video_id, title=excluded.title,
                    channel_title=excluded.channel_title, channel_id=excluded.channel_id,
                    published_at=excluded.published_at, duration_seconds=excluded.duration_seconds,
                    duration_iso8601=excluded.duration_iso8601, definition=excluded.definition,
                    caption=excluded.caption, view_count=excluded.view_count,
                    like_count=excluded.like_count, comment_count=excluded.comment_count,
                    query_hit=excluded.query_hit, tags=excluded.tags, updated_at=excluded.updated_at
                """,
                (
                    source_url, technique, angle, (row.get("video_id") or "").strip(),
                    (row.get("title") or "").strip(), (row.get("channel_title") or "").strip(),
                    (row.get("channel_id") or "").strip(), (row.get("published_at") or "").strip(),
                    int_or_none(row.get("duration_seconds")), (row.get("duration_iso8601") or "").strip(),
                    (row.get("definition") or "").strip(), (row.get("caption") or "").strip(),
                    int_or_none(row.get("view_count")), int_or_none(row.get("like_count")),
                    int_or_none(row.get("comment_count")), (row.get("query_hit") or "").strip(),
                    (row.get("tags") or "").strip(), timestamp, timestamp,
                ),
            )
            imported += 1
    connection.commit()
    return imported


def review_candidate(
    connection: sqlite3.Connection,
    technique: str,
    angle: str,
    source_url: str,
    status: str,
    notes: str = "",
) -> bool:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {status}")
    cursor = connection.execute(
        """
        UPDATE candidates
        SET review_status=?, reviewer_notes=?, updated_at=?
        WHERE source_url=? AND technique=? AND angle=?
        """,
        (
            status, notes, now(), source_url.strip(), normalize_technique_key(technique),
            normalize_technique_key(angle),
        ),
    )
    connection.commit()
    return cursor.rowcount == 1


def list_candidates(
    connection: sqlite3.Connection, technique: str = "", angle: str = "", status: str = ""
) -> list[sqlite3.Row]:
    clauses: list[str] = []
    values: list[str] = []
    if technique:
        clauses.append("technique=?")
        values.append(normalize_technique_key(technique))
    if angle:
        clauses.append("angle=?")
        values.append(normalize_technique_key(angle))
    if status:
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {status}")
        clauses.append("review_status=?")
        values.append(status)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return connection.execute(
        "SELECT * FROM candidates" + where
        + " ORDER BY technique, angle, pose_score IS NULL, pose_score DESC, view_count DESC, source_url",
        values,
    ).fetchall()


def approved_plan_rows(connection: sqlite3.Connection, min_sources: int = 4) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for row in list_candidates(connection, status="approved"):
        grouped.setdefault((row["technique"], row["angle"]), []).append(row)

    output: list[dict[str, str]] = []
    for (technique, angle), candidates in grouped.items():
        urls = list(dict.fromkeys(row["source_url"] for row in candidates))[:4]
        if len(urls) < min_sources:
            continue
        scores = [row["pose_score"] for row in candidates if row["pose_score"] is not None]
        notes = f"approved catalog candidates: {len(urls)}"
        if scores:
            notes += f"; best pose score: {max(scores):.2f}"
        output.append(create_csv_template_row(technique, angle, urls, notes=notes))
    return output


def write_plan(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=DEFAULT_DB, help=f"SQLite catalog path (default: {DEFAULT_DB})")
    commands = parser.add_subparsers(dest="command", required=True)

    imported = commands.add_parser("import-scout", help="import or refresh candidates from scout CSV")
    imported.add_argument("--csv", required=True)

    listed = commands.add_parser("list", help="list candidates for review")
    listed.add_argument("--technique", default="")
    listed.add_argument("--angle", default="")
    listed.add_argument("--status", choices=VALID_STATUSES, default="")

    reviewed = commands.add_parser("review", help="set a candidate's review status")
    reviewed.add_argument("--technique", required=True)
    reviewed.add_argument("--angle", required=True)
    reviewed.add_argument("--url", required=True)
    reviewed.add_argument("--status", choices=VALID_STATUSES, required=True)
    reviewed.add_argument("--notes", default="")

    exported = commands.add_parser("export-plan", help="export approved candidates for batch capture")
    exported.add_argument("--output-csv", default="reference_poses/catalog_capture_plan.csv")
    exported.add_argument("--min-sources", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "export-plan" and not 1 <= args.min_sources <= 4:
        print("min-sources must be between 1 and 4", file=sys.stderr)
        return 2
    connection = connect(project_path(args.db))
    try:
        if args.command == "import-scout":
            path = project_path(args.csv)
            if not path.exists():
                print(f"candidate CSV not found: {path}", file=sys.stderr)
                return 2
            print(f"imported {import_scout_csv(connection, path)} candidate row(s)")
        elif args.command == "list":
            for row in list_candidates(connection, args.technique, args.angle, args.status):
                print(f"{row['review_status']:8} {row['technique']:20} {row['angle']:12} {row['source_url']}\t{row['title']}")
        elif args.command == "review":
            if not review_candidate(connection, args.technique, args.angle, args.url, args.status, args.notes):
                print("candidate not found; import it before reviewing", file=sys.stderr)
                return 2
            print(f"marked candidate {args.status}")
        else:
            rows = approved_plan_rows(connection, args.min_sources)
            output = project_path(args.output_csv)
            write_plan(rows, output)
            print(f"wrote {len(rows)} ready plan row(s) to {output}")
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
