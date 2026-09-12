"""Build the karate reference capture plan from the technique catalogue.

Reads ``reference_poses/karate_techniques.csv`` and writes one plan row per
(technique, angle) in the same 17-column schema
``run_reference_collection_batch.py`` already consumes, so the karate plan can
be run with::

    python scripts/run_reference_collection_batch.py --plan-csv reference_poses/karate_capture_plan.csv

Rows start as ``pending`` with the angle-specific YouTube query recorded in
``notes``, matching how the unsourced rows of the kickboxing plan are stored.
Pass ``--candidates-csv`` to fill source URLs from a reviewed candidate file
(the schema ``scrape_jab_candidates.py`` writes): rows that pick up at least one
URL are promoted to ``ready`` with a runnable capture command.

Usage:
    python scripts/generate_karate_capture_plan.py
    python scripts/generate_karate_capture_plan.py --tier all --candidates-csv reference_poses/karate_video_candidates.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scout_utils import angle_to_camera_description, generate_search_queries, normalize_technique_key
from technique_catalog import techniques

#: Camera angles the reference library is organized by, matching the layout described in
#: reference_poses/README.md and the angles used by the existing all-labels plan.
DEFAULT_ANGLES = ("front", "left45", "right45", "side", "side_left", "side_right", "behind")

PLAN_FIELDNAMES = [
    "technique",
    "angle",
    "reference_key",
    "status",
    "source_url",
    "source_url_1",
    "source_url_2",
    "source_url_3",
    "source_url_4",
    "segment_start_s",
    "segment_end_s",
    "quality",
    "notes",
    "target_technique_key",
    "record_reference_key",
    "command_ready",
    "command",
]

#: Window length per capture profile, mirroring the profiles in
#: run_reference_collection_batch.py so a generated command matches how the batch
#: runner would capture the same technique.
PROFILE_SAMPLE_COUNTS = {"stance": 24, "punch": 20, "kick": 24}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--catalog-csv",
        type=str,
        default="reference_poses/karate_techniques.csv",
        help="technique catalogue to expand (default: reference_poses/karate_techniques.csv)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="reference_poses/karate_capture_plan.csv",
        help="plan file to write (default: reference_poses/karate_capture_plan.csv)",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="core",
        choices=["core", "extended", "all"],
        help="which catalogue tier to include (default: core)",
    )
    parser.add_argument(
        "--angles",
        nargs="+",
        default=list(DEFAULT_ANGLES),
        help=f"camera angles to plan for (default: {' '.join(DEFAULT_ANGLES)})",
    )
    parser.add_argument(
        "--candidates-csv",
        type=str,
        default=None,
        help=(
            "optional reviewed candidate CSV (technique/angle/url columns) used to fill source URLs. "
            "Only rows marked keep=yes are used, and a candidate with a blank angle applies to every "
            "angle of its technique"
        ),
    )
    parser.add_argument(
        "--include-unreviewed",
        action="store_true",
        help=(
            "also use candidate rows whose keep column is still blank. The shipped karate candidate "
            "pool was collected by search and nobody has checked that each clip really shows the "
            "labelled technique from the labelled angle, so this trades correctness for speed"
        ),
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo26n-pose.pt",
        help="YOLO pose weights to reference in generated commands",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default="python",
        help="python executable to use in generated commands",
    )
    return parser.parse_args()


def _resolve_project_path(path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _ps_quote(text: str) -> str:
    return '"' + str(text).replace('"', '`"') + '"'


def _truthy(value: str) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y", "keep"}


def _falsy(value: str) -> bool:
    return (value or "").strip().lower() in {"0", "false", "no", "n", "drop", "skip"}


def build_command(
    python_exe: str,
    weights: str,
    source_url: str,
    record_reference_key: str,
    target_technique_key: str,
    num_video_sequence_samples: int,
) -> str:
    """PowerShell one-liner that captures a single reference window from one clip.

    Mirrors the commands stored in reference_poses/generated_capture_plan_all_labels.csv
    so both plans can be driven by the same batch runner.
    """
    parts = [
        f"& {_ps_quote(python_exe)}",
        _ps_quote("action_recognition.py"),
        "--weights",
        _ps_quote(weights),
        "--source",
        _ps_quote(source_url),
        "--record-reference",
        _ps_quote(record_reference_key),
        "--target-technique",
        _ps_quote(target_technique_key),
        "--reference-dir",
        _ps_quote("reference_poses"),
        "--num-video-sequence-samples",
        str(num_video_sequence_samples),
        "--skip-frame",
        "1",
        "--save-kpts-dir",
        _ps_quote("keypoints"),
        "--record-reference-max-saves",
        "1",
        "--reference-capture-cooldown-frames",
        "24",
        "--disable-video-classifier",
        "--visualize-pose",
        "--auto-exit-after-reference",
        "--reference-search-max-frames",
        "1800",
        "--person-selection-mode",
        _ps_quote("most_motion"),
        "--disable-structured-storage",
        "--reference-capture-mode",
        _ps_quote("best_window"),
        "--ref-min-motion-energy",
        "0.02",
        "--ref-min-score-gate",
        "0",
    ]
    return " ".join(parts)


def load_candidates(path: Path, include_unreviewed: bool = False) -> dict[tuple[str, str], list[dict[str, str]]]:
    """Group reviewed candidate rows by (technique_key, angle).

    Only ``keep=yes`` rows count unless ``include_unreviewed`` is set, in which
    case rows with a blank keep column are used too (an explicit no is always
    honoured). An empty angle means the candidate was not tied to a camera angle
    during review, so it is filed under ``""`` and used as a fallback for any angle.
    """
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    if not path.exists():
        return grouped

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            url = (row.get("url") or "").strip()
            keep = row.get("keep", "")
            if not url or _falsy(keep):
                continue
            if not _truthy(keep) and not include_unreviewed:
                continue
            technique = normalize_technique_key(row.get("technique", ""))
            angle = normalize_technique_key(row.get("angle", ""))
            if technique:
                grouped[(technique, angle)].append(row)
    return grouped


def candidate_urls_for(
    grouped: dict[tuple[str, str], list[dict[str, str]]], technique: str, angle: str, limit: int = 4
) -> list[dict[str, str]]:
    """Angle-specific candidates first, then technique-level ones, up to ``limit``."""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for key in ((technique, angle), (technique, "")):
        for row in grouped.get(key, []):
            url = (row.get("url") or "").strip()
            if url not in seen:
                seen.add(url)
                rows.append(row)
    return rows[:limit]


def main() -> int:
    args = parse_args()

    catalog_path = _resolve_project_path(args.catalog_csv)
    if not catalog_path.exists():
        print(f"error: catalogue not found: {catalog_path}")
        return 2

    tier = None if args.tier == "all" else args.tier
    # Read --catalog-csv itself rather than the merged karate+mma default, so
    # this can target either catalogue standalone (e.g. mma_techniques.csv).
    entries = techniques(tier=tier, path=catalog_path)
    if not entries:
        print(f"error: no techniques with tier={args.tier} in {catalog_path}")
        return 2

    grouped = (
        load_candidates(_resolve_project_path(args.candidates_csv), include_unreviewed=args.include_unreviewed)
        if args.candidates_csv
        else {}
    )

    rows: list[dict[str, str]] = []
    ready = 0
    for entry in entries:
        technique = entry["technique_key"]
        samples = PROFILE_SAMPLE_COUNTS.get(entry.get("capture_profile", ""), 20)
        for angle in args.angles:
            reference_key = f"{technique}__{angle}"
            queries = generate_search_queries(technique, angle, num_queries=1)
            query = queries[0] if queries else f"{technique} {angle_to_camera_description(angle)}"

            candidates = candidate_urls_for(grouped, technique, angle)
            urls = [(c.get("url") or "").strip() for c in candidates]
            padded = (urls + ["", "", "", ""])[:4]

            row = {
                "technique": technique,
                "angle": angle,
                "reference_key": reference_key,
                "status": "ready" if urls else "pending",
                "source_url": padded[0],
                "source_url_1": padded[0],
                "source_url_2": padded[1],
                "source_url_3": padded[2],
                "source_url_4": padded[3],
                "segment_start_s": (candidates[0].get("segment_start_s", "") if candidates else ""),
                "segment_end_s": (candidates[0].get("segment_end_s", "") if candidates else ""),
                "quality": f"auto: {query}" if urls else "",
                "notes": (
                    f"{entry['romaji']} ({entry['japanese']}) - {entry['english']}"
                    if urls
                    else f"needs source url: {query}"
                ),
                "target_technique_key": technique,
                "record_reference_key": reference_key,
                "command_ready": "yes" if urls else "no",
                "command": (
                    build_command(
                        python_exe=args.python_exe,
                        weights=args.weights,
                        source_url=padded[0],
                        record_reference_key=reference_key,
                        target_technique_key=technique,
                        num_video_sequence_samples=samples,
                    )
                    if urls
                    else ""
                ),
            }
            ready += 1 if urls else 0
            rows.append(row)

    out_path = _resolve_project_path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PLAN_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} plan rows ({len(entries)} techniques x {len(args.angles)} angles) to {out_path}")
    print(f"  ready:   {ready}")
    print(f"  pending: {len(rows) - ready}")
    if ready < len(rows):
        print("  fill pending rows with scripts/scout_youtube_by_golden_seeds.py or a reviewed --candidates-csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
