"""Scout YouTube videos using Golden Seeds as templates for batch reference collection.

This script:
1. Scans Golden Seeds directory for techniques and camera angles
2. Generates YouTube search queries for each technique/angle
3. Searches YouTube API for candidate videos
4. Ranks candidates and populates CSV plan for batch collection
5. Outputs candidates CSV for manual review before running batch collection

Usage:
    python scout_youtube_by_golden_seeds.py --api-key YOUR_KEY
    python scout_youtube_by_golden_seeds.py --api-key YOUR_KEY --technique jab --angle side_right
    python scout_youtube_by_golden_seeds.py --api-key YOUR_KEY --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import ssl
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from scout_utils import (
    generate_search_queries,
    inventory_golden_seeds,
    create_csv_template_row,
    infer_angle_from_filename,
    normalize_technique_key,
)

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _resolve_project_path(path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scout YouTube videos for each Golden Seeds technique/angle and populate "
            "reference collection batch plan CSV."
        )
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("YOUTUBE_API_KEY", ""),
        help="YouTube Data API v3 key. Defaults to env var YOUTUBE_API_KEY.",
    )
    parser.add_argument(
        "--technique",
        type=str,
        default="",
        help="Optional: limit to one technique (e.g., 'jab', 'fighting_stance'). Scans all if not specified.",
    )
    parser.add_argument(
        "--angle",
        type=str,
        default="",
        help="Optional: limit to one angle within technique. Requires --technique.",
    )
    parser.add_argument(
        "--max-results-per-query",
        type=int,
        default=50,
        help="Max results per query page (1-50).",
    )
    parser.add_argument(
        "--max-pages-per-query",
        type=int,
        default=2,
        help="Max pages to fetch for each query.",
    )
    parser.add_argument(
        "--max-candidates-per-angle",
        type=int,
        default=4,
        help="Take top N candidates per angle (will populate source_url_1-4).",
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
        "--min-view-count",
        type=int,
        default=1000,
        help="Filter out videos with fewer views.",
    )
    parser.add_argument(
        "--order",
        type=str,
        default="relevance",
        choices=["date", "rating", "relevance", "title", "videoCount", "viewCount"],
        help="YouTube search ordering.",
    )
    parser.add_argument(
        "--output-candidates-csv",
        type=str,
        default="reference_poses/scout_candidates_golden_seeds.csv",
        help="Output detailed candidates CSV for manual review.",
    )
    parser.add_argument(
        "--output-plan-csv",
        type=str,
        default="reference_poses/scout_batch_plan.csv",
        help="Output batch plan CSV ready for run_reference_collection_batch.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned queries without making API calls.",
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
        help="Disable TLS certificate verification (local environments only).",
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


def parse_iso8601_duration_to_seconds(duration: str) -> int:
    """Parse ISO8601 duration (PT1H2M3S) to seconds."""
    import re

    m = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$", duration)
    if not m:
        return 0
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    seconds = int(float(m.group(3) or 0))
    return hours * 3600 + minutes * 60 + seconds


def coerce_int(s: str | int, default: int = 0) -> int:
    try:
        return int(s)
    except (ValueError, TypeError):
        return default


def search_videos_for_query(
    args: argparse.Namespace,
    api_key: str,
    query: str,
    tls_context: ssl.SSLContext | None,
) -> list[dict[str, Any]]:
    """Search YouTube for a query and return metadata."""
    videos: list[dict[str, Any]] = []
    page_token = ""

    for page_num in range(max(args.max_pages_per_query, 1)):
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max(1, min(args.max_results_per_query, 50)),
            "order": args.order,
            "pageToken": page_token,
        }
        
        try:
            payload = youtube_get("search", api_key, params, tls_context=tls_context)
        except Exception as exc:
            print(f"  warning: search failed for query '{query}': {exc}")
            break

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
    """Fetch detailed metadata for a batch of video IDs."""
    out: dict[str, dict[str, Any]] = {}
    
    for i in range(0, len(video_ids), 50):
        group = video_ids[i : i + 50]
        try:
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
        except Exception as exc:
            print(f"  warning: fetch_video_details failed: {exc}")
            continue

        for item in payload.get("items", []):
            vid = item.get("id", "")
            if not vid:
                continue
            out[vid] = {
                "duration_iso8601": item.get("contentDetails", {}).get("duration", ""),
                "definition": item.get("contentDetails", {}).get("definition", ""),
                "caption": item.get("contentDetails", {}).get("caption", ""),
                "view_count": coerce_int(item.get("statistics", {}).get("viewCount", "0"), 0),
                "like_count": coerce_int(item.get("statistics", {}).get("likeCount", "0"), 0),
                "comment_count": coerce_int(item.get("statistics", {}).get("commentCount", "0"), 0),
                "channel_id": item.get("snippet", {}).get("channelId", ""),
                "tags": "|".join(item.get("snippet", {}).get("tags", [])[:20]),
            }
    return out


def scout_technique_angle(
    args: argparse.Namespace,
    api_key: str,
    technique: str,
    angle: str,
    tls_context: ssl.SSLContext | None,
) -> list[dict[str, Any]]:
    """Scout YouTube for a specific technique/angle pair."""
    queries = generate_search_queries(technique, angle, num_queries=5)
    print(f"\n  {technique:20} / {angle:12} -> {len(queries)} queries")
    
    all_videos: list[dict[str, Any]] = []
    for query in queries:
        print(f"    query: {query[:60]}...")
        videos = search_videos_for_query(args, api_key, query, tls_context=tls_context)
        print(f"      -> {len(videos)} hits")
        all_videos.extend(videos)
    
    # Deduplicate by video_id
    by_id: dict[str, dict[str, Any]] = {}
    for video in all_videos:
        vid = video["video_id"]
        if vid not in by_id:
            by_id[vid] = video
        else:
            # Merge query_hit list
            prev = by_id[vid]
            qset = {
                q.strip()
                for q in (prev.get("query_hit", "") + "|" + video.get("query_hit", "")).split("|")
                if q.strip()
            }
            prev["query_hit"] = "|".join(sorted(qset))

    deduped = list(by_id.values())
    print(f"    deduplicated to {len(deduped)} unique videos")
    
    # Fetch detailed metrics
    video_ids = [v["video_id"] for v in deduped]
    details = fetch_video_details(api_key, video_ids, tls_context=tls_context)
    
    # Filter and rank
    final: list[dict[str, Any]] = []
    for video in deduped:
        d = details.get(video["video_id"], {})
        duration_iso = d.get("duration_iso8601", "")
        duration_s = parse_iso8601_duration_to_seconds(duration_iso)
        view_count = d.get("view_count", 0)
        
        if duration_s < args.min_duration_seconds or duration_s > args.max_duration_seconds:
            continue
        if view_count < args.min_view_count:
            continue
        
        final.append(
            {
                "technique": technique,
                "angle": angle,
                "video_id": video["video_id"],
                "url": f"https://www.youtube.com/watch?v={video['video_id']}",
                "title": video.get("title", ""),
                "channel_title": video.get("channel_title", ""),
                "channel_id": d.get("channel_id", ""),
                "published_at": video.get("published_at", ""),
                "duration_iso8601": duration_iso,
                "duration_seconds": duration_s,
                "definition": d.get("definition", ""),
                "caption": d.get("caption", ""),
                "view_count": view_count,
                "like_count": d.get("like_count", 0),
                "comment_count": d.get("comment_count", 0),
                "query_hit": video.get("query_hit", ""),
                "tags": d.get("tags", ""),
            }
        )
    
    # Rank by view count descending
    final.sort(key=lambda r: r["view_count"], reverse=True)
    return final


def main() -> int:
    args = parse_args()
    
    if not args.api_key and not args.dry_run:
        print("error: missing API key. Set --api-key or env var YOUTUBE_API_KEY.", file=sys.stderr)
        return 2
    
    project_root = PROJECT_ROOT
    golden_dir = project_root / "reference_poses" / "Golden_Seeds"
    
    print("=" * 80)
    print("GOLDEN SEEDS YOUTUBE SCOUT")
    print("=" * 80)
    
    # Scan Golden Seeds
    print(f"\nScanning Golden Seeds: {golden_dir}")
    inventory = inventory_golden_seeds(golden_dir)
    if not inventory:
        print(f"error: no Golden Seeds found in {golden_dir}")
        return 2
    
    print(f"Found {len(inventory)} technique(s):")
    for tech in sorted(inventory.keys()):
        angles = inventory[tech]
        print(f"  {tech}: {len(angles)} angle(s) - {', '.join(sorted(angles.keys()))}")
    
    # Build list of (technique, angle) to scout
    scout_targets: list[tuple[str, str]] = []
    if args.technique:
        technique_lower = normalize_technique_key(args.technique)
        if technique_lower in inventory:
            if args.angle:
                if args.angle in inventory[technique_lower]:
                    scout_targets.append((technique_lower, args.angle))
                else:
                    print(f"error: angle '{args.angle}' not found in '{technique_lower}'")
                    return 2
            else:
                for angle in inventory[technique_lower]:
                    scout_targets.append((technique_lower, angle))
        else:
            print(f"error: technique '{technique_lower}' not found in Golden Seeds")
            return 2
    else:
        for tech in inventory:
            for angle in inventory[tech]:
                scout_targets.append((tech, angle))
    
    print(f"\nScout targets: {len(scout_targets)} combinations")
    for tech, angle in scout_targets:
        queries = generate_search_queries(tech, angle, num_queries=1)  # Just show first query
        print(f"  {tech:20} / {angle:12} -> {queries[0][:50]}...")
    
    if args.dry_run:
        print("\n[DRY RUN] Exiting before API calls.")
        return 0
    
    # Setup TLS context
    tls_context: ssl.SSLContext | None = None
    if args.ca_bundle:
        tls_context = ssl.create_default_context(cafile=args.ca_bundle)
    elif args.insecure_skip_tls_verify:
        tls_context = ssl.create_default_context()
        tls_context.check_hostname = False
        tls_context.verify_mode = ssl.CERT_NONE
    
    # Scout each target
    print("\n" + "=" * 80)
    print("SEARCHING YOUTUBE")
    print("=" * 80)
    
    all_candidates: list[dict[str, Any]] = []
    batch_plan_rows: list[dict[str, str]] = []
    
    for technique, angle in scout_targets:
        try:
            results = scout_technique_angle(args, args.api_key, technique, angle, tls_context)
            all_candidates.extend(results)
            
            # Take top N for batch plan
            top_urls = [r["url"] for r in results[: args.max_candidates_per_angle]]
            if top_urls:
                row = create_csv_template_row(
                    technique=technique,
                    angle=angle,
                    source_urls=top_urls,
                    notes=f"auto from scout: {len(results)} candidates found",
                )
                batch_plan_rows.append(row)
                print(f"    -> added {len(top_urls)} URL(s) to batch plan")
        except Exception as exc:
            print(f"  error scouting {technique}/{angle}: {exc}")
            continue
    
    # Write candidates CSV
    print("\n" + "=" * 80)
    print("WRITING OUTPUT")
    print("=" * 80)
    
    candidates_path = _resolve_project_path(args.output_candidates_csv)
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    
    if all_candidates:
        fieldnames = [
            "technique",
            "angle",
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
        
        with candidates_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_candidates)
        
        print(f"✓ Wrote {len(all_candidates)} candidates to: {candidates_path}")
    
    # Write batch plan CSV
    plan_path = _resolve_project_path(args.output_plan_csv)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    
    if batch_plan_rows:
        fieldnames = list(batch_plan_rows[0].keys())
        with plan_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(batch_plan_rows)
        
        print(f"✓ Wrote {len(batch_plan_rows)} batch plan rows to: {plan_path}")
        print(f"\nNext steps:")
        print(f"  1. Review candidates: {candidates_path}")
        print(f"  2. Copy/merge to: reference_poses/generated_capture_plan_all_labels.csv")
        print(f"  3. Run: python run_reference_collection_batch.py")
    
    now = datetime.now(timezone.utc).isoformat()
    print(f"\nCompleted at: {now}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
