from __future__ import annotations

import argparse
import csv
import json
import os
import re
import ssl
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _resolve_project_path(path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else PROJECT_ROOT / p

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"

DEFAULT_QUERIES = [
    "boxing jab tutorial",
    "muay thai jab technique",
    "jab form slow motion",
    "how to throw a jab",
    "proper jab mechanics",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect candidate jab-technique YouTube videos into a CSV for manual review. "
            "This script only collects metadata and URLs."
        )
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("YOUTUBE_API_KEY", ""),
        help="YouTube Data API v3 key. Defaults to env var YOUTUBE_API_KEY.",
    )
    parser.add_argument(
        "--query",
        nargs="+",
        action="append",
        default=None,
        help=(
            "Search query. Can be passed multiple times. "
            "Example: --query boxing jab tutorial --query muay thai jab technique"
        ),
    )
    parser.add_argument(
        "--max-results-per-query",
        type=int,
        default=50,
        help="Max results requested per query page (1-50).",
    )
    parser.add_argument(
        "--max-pages-per-query",
        type=int,
        default=2,
        help="Max pages to fetch for each query.",
    )
    parser.add_argument(
        "--region-code",
        type=str,
        default="",
        help="Optional ISO 3166-1 alpha-2 region code, for example US.",
    )
    parser.add_argument(
        "--relevance-language",
        type=str,
        default="en",
        help="Language hint for search relevance, for example en.",
    )
    parser.add_argument(
        "--published-after",
        type=str,
        default="",
        help="Optional ISO8601 timestamp, for example 2020-01-01T00:00:00Z.",
    )
    parser.add_argument(
        "--published-before",
        type=str,
        default="",
        help="Optional ISO8601 timestamp, for example 2026-01-01T00:00:00Z.",
    )
    parser.add_argument(
        "--min-duration-seconds",
        type=int,
        default=10,
        help="Filter out clips shorter than this many seconds.",
    )
    parser.add_argument(
        "--max-duration-seconds",
        type=int,
        default=600,
        help="Filter out clips longer than this many seconds.",
    )
    parser.add_argument(
        "--order",
        type=str,
        default="relevance",
        choices=["date", "rating", "relevance", "title", "videoCount", "viewCount"],
        help="YouTube search ordering.",
    )
    parser.add_argument(
        "--ca-bundle",
        type=str,
        default="",
        help="Optional CA bundle file for TLS verification.",
    )
    parser.add_argument(
        "--insecure-skip-tls-verify",
        action="store_true",
        help="Disable TLS certificate verification (use only in constrained local environments).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="reference_poses/jab_video_candidates.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--technique",
        type=str,
        default="jab",
        help="Technique tag prefilled in the CSV.",
    )
    return parser.parse_args()


def _http_get_json(url: str, tls_context: ssl.SSLContext | None = None) -> dict[str, Any]:
    with urlopen(url, timeout=30, context=tls_context) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def youtube_get(
    endpoint: str,
    api_key: str,
    params: dict[str, Any],
    tls_context: ssl.SSLContext | None = None,
) -> dict[str, Any]:
    p = {k: v for k, v in params.items() if v not in (None, "")}
    p["key"] = api_key
    qs = urlencode(p, doseq=True)
    url = f"{YOUTUBE_API_BASE}/{endpoint}?{qs}"
    return _http_get_json(url, tls_context=tls_context)


def flatten_queries(query_args: list[list[str]] | None) -> list[str]:
    if not query_args:
        return DEFAULT_QUERIES.copy()
    return [" ".join(group).strip() for group in query_args if " ".join(group).strip()]


def parse_iso8601_duration_to_seconds(duration: str) -> int:
    # Supports patterns like PT4M13S, PT58S, PT1H2M, PT1H2M3S
    m = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$", duration)
    if not m:
        return 0
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    seconds = int(float(m.group(3) or 0))
    return hours * 3600 + minutes * 60 + seconds


def chunked(items: list[str], n: int) -> list[list[str]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def search_videos_for_query(
    args: argparse.Namespace,
    api_key: str,
    query: str,
    tls_context: ssl.SSLContext | None,
) -> list[dict[str, Any]]:
    videos: list[dict[str, Any]] = []
    page_token = ""

    for _ in range(max(args.max_pages_per_query, 1)):
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max(1, min(args.max_results_per_query, 50)),
            "order": args.order,
            "regionCode": args.region_code,
            "relevanceLanguage": args.relevance_language,
            "publishedAfter": args.published_after,
            "publishedBefore": args.published_before,
            "pageToken": page_token,
        }
        payload = youtube_get("search", api_key, params, tls_context=tls_context)

        for item in payload.get("items", []):
            video_id = item.get("id", {}).get("videoId", "")
            snippet = item.get("snippet", {})
            if not video_id:
                continue
            videos.append(
                {
                    "video_id": video_id,
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", ""),
                    "channel_title": snippet.get("channelTitle", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "query_hit": query,
                }
            )

        page_token = payload.get("nextPageToken", "")
        if not page_token:
            break

    return videos


def fetch_video_details(
    api_key: str,
    video_ids: list[str],
    tls_context: ssl.SSLContext | None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for group in chunked(video_ids, 50):
        payload = youtube_get(
            "videos",
            api_key,
            {
                "part": "snippet,contentDetails,statistics",
                "id": ",".join(group),
                "maxResults": 50,
            },
            tls_context=tls_context,
        )

        for item in payload.get("items", []):
            vid = item.get("id", "")
            if not vid:
                continue
            out[vid] = {
                "duration_iso8601": item.get("contentDetails", {}).get("duration", ""),
                "definition": item.get("contentDetails", {}).get("definition", ""),
                "caption": item.get("contentDetails", {}).get("caption", ""),
                "view_count": item.get("statistics", {}).get("viewCount", "0"),
                "like_count": item.get("statistics", {}).get("likeCount", "0"),
                "comment_count": item.get("statistics", {}).get("commentCount", "0"),
                "channel_id": item.get("snippet", {}).get("channelId", ""),
                "tags": "|".join(item.get("snippet", {}).get("tags", [])[:20]),
            }
    return out


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        vid = row["video_id"]
        if vid not in by_id:
            by_id[vid] = row
            continue

        # Keep a merged query_hit list for traceability.
        prev = by_id[vid]
        qset = {q.strip() for q in (prev.get("query_hit", "") + "|" + row.get("query_hit", "")).split("|") if q.strip()}
        prev["query_hit"] = "|".join(sorted(qset))
        by_id[vid] = prev

    return list(by_id.values())


def coerce_int(s: str, default: int = 0) -> int:
    try:
        return int(s)
    except (ValueError, TypeError):
        return default


def main() -> int:
    args = parse_args()

    if not args.api_key:
        print("error: missing API key. Set --api-key or env var YOUTUBE_API_KEY.", file=sys.stderr)
        return 2

    queries = flatten_queries(args.query)
    if not queries:
        print("error: no queries were provided.", file=sys.stderr)
        return 2

    print(f"collecting candidate videos for technique '{args.technique}'...")
    print(f"queries: {queries}")

    tls_context: ssl.SSLContext | None = None
    if args.ca_bundle:
        tls_context = ssl.create_default_context(cafile=args.ca_bundle)
    elif args.insecure_skip_tls_verify:
        tls_context = ssl.create_default_context()
        tls_context.check_hostname = False
        tls_context.verify_mode = ssl.CERT_NONE

    all_rows: list[dict[str, Any]] = []
    for q in queries:
        rows = search_videos_for_query(args, args.api_key, q, tls_context=tls_context)
        print(f"query '{q}' -> {len(rows)} hits")
        all_rows.extend(rows)

    deduped = deduplicate(all_rows)
    print(f"deduplicated to {len(deduped)} unique videos")

    details = fetch_video_details(args.api_key, [r["video_id"] for r in deduped], tls_context=tls_context)

    final_rows: list[dict[str, Any]] = []
    for row in deduped:
        d = details.get(row["video_id"], {})
        duration_iso = d.get("duration_iso8601", "")
        duration_s = parse_iso8601_duration_to_seconds(duration_iso)

        if duration_s < args.min_duration_seconds or duration_s > args.max_duration_seconds:
            continue

        final_rows.append(
            {
                "technique": args.technique,
                "angle": "",  # fill manually: front, left45, right45, side, side_right, side_left, behind
                "stance": "",  # fill manually: orthodox, southpaw
                "quality": "",  # fill manually: good, ok, bad
                "keep": "",  # fill manually: yes/no
                "segment_start_s": "",  # fill manually
                "segment_end_s": "",  # fill manually
                "notes": "",  # fill manually
                "video_id": row["video_id"],
                "url": f"https://www.youtube.com/watch?v={row['video_id']}",
                "title": row.get("title", ""),
                "channel_title": row.get("channel_title", ""),
                "channel_id": d.get("channel_id", ""),
                "published_at": row.get("published_at", ""),
                "duration_iso8601": duration_iso,
                "duration_seconds": duration_s,
                "definition": d.get("definition", ""),
                "caption": d.get("caption", ""),
                "view_count": coerce_int(d.get("view_count", "0"), 0),
                "like_count": coerce_int(d.get("like_count", "0"), 0),
                "comment_count": coerce_int(d.get("comment_count", "0"), 0),
                "query_hit": row.get("query_hit", ""),
                "tags": d.get("tags", ""),
            }
        )

    final_rows.sort(key=lambda r: (r["view_count"], r["published_at"]), reverse=True)

    out_path = _resolve_project_path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "technique",
        "angle",
        "stance",
        "quality",
        "keep",
        "segment_start_s",
        "segment_end_s",
        "notes",
        "video_id",
        "url",
        "title",
        "channel_title",
        "channel_id",
        "published_at",
        "duration_iso8601",
        "duration_seconds",
        "definition",
        "caption",
        "view_count",
        "like_count",
        "comment_count",
        "query_hit",
        "tags",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    now = datetime.now(timezone.utc).isoformat()
    print(f"wrote {len(final_rows)} candidates to {out_path}")
    print(f"completed_at_utc={now}")
    print("next step: manually review CSV and fill angle/quality/segment columns.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
