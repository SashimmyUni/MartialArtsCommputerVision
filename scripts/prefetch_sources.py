"""Download every source video a capture plan names, once, before capturing.

Capture used to stream each YouTube source straight into ``cv2.VideoCapture``
every time it was needed. Two problems with that:

- The 52 ready plan rows name 119 distinct URLs across 208 example slots, and
  60 of those URLs appear in more than one row, so the same video was fetched
  and decoded several times over.
- A stream interruption kills a long run. ``docs/HOWTO.md`` §9.4 records this
  as a live problem with overnight automation and asks for a pre-download
  policy; this is that policy.

Prefetching first makes the expensive part of a batch local and restartable:
the download either succeeded once, or it is retried here rather than in the
middle of a capture.

Usage::

    python scripts/prefetch_sources.py                    # whole ready plan
    python scripts/prefetch_sources.py --dry-run          # list what it would fetch
    python scripts/prefetch_sources.py --technique jab    # one technique
    python scripts/prefetch_sources.py --limit 10         # first 10 missing

Videos land in ``cache/videos/<video_id>.<ext>`` (gitignored). Nothing here is
committed — only the derived keypoint arrays under ``reference_poses/`` are.

Note on segments: ``segment_start_s``/``segment_end_s`` in the plan are *not*
applied at download time. Trimming the download would give the same video a
different cache entry per row, fragmenting the cache precisely for the URLs
shared across rows. The full video is cached once and the segment is applied
later, as a frame range during extraction, where it costs nothing and can be
changed without re-downloading.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import reference_cache as rc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-download every source video named by the capture plan into cache/videos/."
    )
    parser.add_argument(
        "--plan",
        default="reference_poses/generated_capture_plan_all_labels.csv",
        help="capture plan CSV (default: reference_poses/generated_capture_plan_all_labels.csv)",
    )
    parser.add_argument(
        "--technique",
        default=None,
        help="only fetch sources for this technique (default: all)",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="include rows that are not marked command_ready=yes",
    )
    parser.add_argument(
        "--format",
        default=rc.DEFAULT_FORMAT,
        help=f"yt-dlp format selector (default: {rc.DEFAULT_FORMAT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="stop after this many downloads (0 = no limit)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download sources that are already cached",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be fetched without downloading",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    plan_path = Path(args.plan)
    if not plan_path.is_absolute():
        plan_path = rc.PROJECT_ROOT / plan_path
    if not plan_path.exists():
        print(f"plan file not found: {plan_path}")
        return 2

    rows = rc.plan_rows(plan_path, ready_only=not args.all_rows)
    if args.technique:
        wanted = args.technique.strip().lower()
        rows = [r for r in rows if (r.get("technique") or "").strip().lower() == wanted]

    # One entry per video id, remembering which rows wanted it — that mapping is
    # the whole point, so the report can show what the dedup actually saved.
    wanted_by_id: dict[str, dict[str, object]] = {}
    for row in rows:
        label = f"{(row.get('technique') or '?').strip()}/{(row.get('angle') or '?').strip()}"
        for url in rc.collect_source_urls(row):
            video_id = rc.video_id_for_source(url)
            entry = wanted_by_id.setdefault(video_id, {"url": url, "rows": []})
            entry["rows"].append(label)  # type: ignore[union-attr]

    total_slots = sum(len(e["rows"]) for e in wanted_by_id.values())  # type: ignore[arg-type]
    already = [vid for vid in wanted_by_id if rc.find_cached_video(vid) is not None]
    missing = [vid for vid in wanted_by_id if vid not in set(already)]

    print(f"plan: {plan_path.relative_to(rc.PROJECT_ROOT)}")
    print(f"rows: {len(rows)}")
    print(f"source slots: {total_slots} -> {len(wanted_by_id)} distinct video(s)")
    shared = sum(1 for e in wanted_by_id.values() if len(e["rows"]) > 1)  # type: ignore[arg-type]
    if shared:
        print(f"  {shared} video(s) are used by more than one row (fetched once, not per row)")
    print(f"cached already: {len(already)}")
    print(f"to fetch: {len(missing)}")

    targets = missing if not args.force else list(wanted_by_id)
    if args.limit > 0:
        targets = targets[: args.limit]

    if args.dry_run:
        for vid in targets:
            entry = wanted_by_id[vid]
            print(f"  would fetch {vid}  {entry['url']}  (rows: {', '.join(entry['rows'][:4])})")  # type: ignore[index]
        return 0

    if not targets:
        print("nothing to do")
        return 0

    print(f"cache dir: {rc.VIDEO_CACHE_DIR}")
    fetched = 0
    failed: list[tuple[str, str]] = []
    for i, vid in enumerate(targets, start=1):
        entry = wanted_by_id[vid]
        url = str(entry["url"])
        print(f"[{i}/{len(targets)}] {vid}  {url}")
        path = rc.ensure_local_video(url, fmt=args.format, force=args.force)
        if path is None:
            failed.append((vid, url))
            continue
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  -> {path.name} ({size_mb:.1f} MB)")
        fetched += 1

    print(
        "summary:",
        {
            "fetched": fetched,
            "failed": len(failed),
            "already_cached": len(already),
            "distinct_videos": len(wanted_by_id),
        },
    )
    if failed:
        print("failed sources (capture will fall back to streaming for these):")
        for vid, url in failed:
            print(f"  {vid}  {url}")
    # A failed download is not a batch failure: capture still streams.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
