"""Rank scouted candidate videos by what the pose model actually saw in them.

The scout ranks YouTube candidates by view count and `filter_candidates.py`
ranks them by title and tag keywords. Neither looks at the video. A popular,
well-titled clip can still be a talking-head breakdown with no clean technique
in it, and that only becomes apparent after a capture run produces a weak
reference and someone reviews it by hand.

`scout_utils.compute_pose_match_score` was written for this and never wired up.
It is not what this uses: its 30% sequence-length term rewards windows that
merely happen to be the same length, which is meaningless for variable-length
`stance_cycle` output, and it is blind to mirroring and joint angles.
`action_recognition._best_reference_match` — the same function the trainer
scores with — handles all three.

This is a prescreen, not a free lunch: a candidate has to be downloaded and run
through the pose model before it can be scored. It is worth it when that cost
is smaller than a full capture plus human review of the references that follow,
and it is cheaper than it used to be because `extract_tracks.py` caches the
result, so a candidate that survives the screen is not extracted again.

Usage::

    # Extract the candidates you want to screen (once; cached afterwards)
    python scripts/extract_tracks.py --videos <url> <url> ... --max-frames 600

    # Rank them against the reference bank for a technique
    python scripts/rank_candidates_by_pose.py --technique jab --videos <url> <url> ...

    # Or score every candidate the scout CSV lists that is already extracted
    python scripts/rank_candidates_by_pose.py --technique jab \\
        --candidates-csv reference_poses/scout_candidates_golden_seeds.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import action_recognition as ar  # noqa: E402
import reference_cache as rc  # noqa: E402
import reference_selection as rs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score candidate videos by pose similarity to a technique's reference bank."
    )
    parser.add_argument("--technique", required=True, help="technique to score against, e.g. jab")
    parser.add_argument("--videos", nargs="*", default=[], help="candidate URLs or paths")
    parser.add_argument(
        "--candidates-csv",
        default=None,
        help="scout candidates CSV; every row with a video URL is scored if it has a cached extraction",
    )
    parser.add_argument(
        "--reference-dir",
        default="reference_poses",
        help="bank to score against (default: reference_poses). Point at a Golden Seed bank to "
             "rank candidates by resemblance to your own recordings",
    )
    parser.add_argument("--top", type=int, default=20, help="rows to print (default: 20)")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="only report candidates scoring at least this (default: 0)",
    )
    parser.add_argument("--out-csv", default=None, help="also write the ranking to this CSV")
    return parser.parse_args()


def _csv_urls(path: Path) -> list[str]:
    """Video URLs from a scout candidates CSV, in file order, deduped."""
    urls: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            for column in ("url", "video_url", "source_url", "watch_url", "link"):
                value = (row.get(column) or "").strip()
                if value and value not in seen:
                    seen.add(value)
                    urls.append(value)
                    break
    return urls


def score_candidate_video(
    video_id: str,
    technique_key: str,
    bank: dict[str, "object"],
    config: rs.SelectionConfig,
) -> dict[str, object] | None:
    """Best window the cached extraction offers, scored against ``bank``.

    Scores the *best available window* rather than the whole clip: that is what
    capture would actually take from this video, so it is the honest estimate of
    what the candidate is worth.
    """
    cache = rc.load_best_track_cache(video_id, needed_frames=config.search_max_frames)
    if cache is None:
        return None

    candidates = rs.iter_candidates(cache, config)
    if not candidates:
        return {
            "video_id": video_id,
            "score": 0.0,
            "windows": 0,
            "note": "no window passed the motion/closure gates",
        }

    best_score = -1.0
    best = None
    for candidate in candidates:
        match = ar._best_reference_match(candidate.pose_seq, bank, technique_key)
        if match is None:
            continue
        score = float(match[1]["score"])
        if score > best_score:
            best_score = score
            best = (candidate, match[0])

    if best is None:
        return {"video_id": video_id, "score": 0.0, "windows": len(candidates), "note": "unscorable"}

    candidate, angle = best
    return {
        "video_id": video_id,
        "score": round(best_score, 2),
        "windows": len(candidates),
        "best_angle": angle,
        "frames": f"{candidate.start_frame}-{candidate.last_frame}",
        "motion": round(candidate.energy, 3),
        "closure": round(candidate.closure, 3),
        "note": "",
    }


def main() -> int:
    args = parse_args()

    reference_root = Path(args.reference_dir)
    if not reference_root.is_absolute():
        reference_root = PROJECT_ROOT / reference_root
    references = ar.load_reference_pose_library(str(reference_root))

    technique_key = ar._normalize_key(args.technique)
    bank = references.get(technique_key)
    if not bank:
        print(f"no references for '{args.technique}' under {reference_root}")
        print(f"available: {', '.join(sorted(references)) or '(none)'}")
        return 2
    print(f"scoring against {len(bank)} reference(s) for {technique_key}")

    sources = list(args.videos)
    if args.candidates_csv:
        csv_path = Path(args.candidates_csv)
        if not csv_path.is_absolute():
            csv_path = PROJECT_ROOT / csv_path
        if not csv_path.exists():
            print(f"candidates CSV not found: {csv_path}")
            return 2
        sources.extend(_csv_urls(csv_path))
    if not sources:
        print("no candidates given (use --videos or --candidates-csv)")
        return 2

    from run_reference_collection_batch import TECHNIQUE_CAPTURE_PROFILES

    profile = TECHNIQUE_CAPTURE_PROFILES.get(technique_key, {})
    config = rs.SelectionConfig.from_profile(dict(profile)) if profile else rs.SelectionConfig()

    by_id: dict[str, str] = {}
    for source in sources:
        by_id.setdefault(rc.video_id_for_source(source), source)

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for video_id, source in by_id.items():
        result = score_candidate_video(video_id, technique_key, bank, config)
        if result is None:
            missing.append(source)
            continue
        result["source"] = source
        rows.append(result)

    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    kept = [r for r in rows if float(r["score"]) >= args.min_score]

    print(f"\nscored {len(rows)} candidate(s); {len(kept)} at or above {args.min_score}")
    print(f"{'score':>7}  {'windows':>7}  {'angle':<14} {'frames':<12} source")
    for row in kept[: args.top]:
        print(
            f"{row['score']:>7}  {row['windows']:>7}  {str(row.get('best_angle', '')):<14} "
            f"{str(row.get('frames', '')):<12} {row['source']}"
            + (f"   [{row['note']}]" if row.get("note") else "")
        )

    if missing:
        print(f"\n{len(missing)} candidate(s) have no cached extraction and were not scored. Run:")
        print("  python scripts/extract_tracks.py --videos " + " ".join(missing[:3]) + (" ..." if len(missing) > 3 else ""))

    if args.out_csv and rows:
        out_path = Path(args.out_csv)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["score", "windows", "best_angle", "frames", "motion", "closure", "video_id", "source", "note"]
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote ranking to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
