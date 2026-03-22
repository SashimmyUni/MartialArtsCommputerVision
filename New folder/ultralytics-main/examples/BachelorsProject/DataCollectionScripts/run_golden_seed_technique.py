from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _resolve_project_path(path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _to_pascal_case(snake_case: str) -> str:
    return "".join(part.capitalize() for part in snake_case.split("_") if part)


def _infer_angle_from_filename(file_name: str) -> str | None:
    text = file_name.lower()

    if "behind" in text or "rear" in text or "back" in text:
        return "behind"
    if "left45" in text or "45left" in text or "45_left" in text or "left_45" in text:
        return "left45"
    if "right45" in text or "45right" in text or "45_right" in text or "right_45" in text:
        return "right45"
    if "side_left" in text or "left_side" in text:
        return "side_left"
    if "side_right" in text or "right_side" in text:
        return "side_right"
    if "front" in text:
        return "front"
    if re.search(r"(^|[_\-.\s])left($|[_\-.\s])", text):
        return "side_left"
    if re.search(r"(^|[_\-.\s])right($|[_\-.\s])", text):
        return "side_right"
    if "side" in text or "profile" in text:
        return "side"
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run all Golden Seeds files for one technique and extract references into "
            "reference_poses/<technique_key>/"
        )
    )
    parser.add_argument(
        "--technique-key",
        required=True,
        help="Technique key used in record-reference and output folder, e.g. fighting_stance",
    )
    parser.add_argument(
        "--golden-technique-dir",
        default="",
        help=(
            "Folder under reference_poses/Golden_Seeds (default: PascalCase from technique key, "
            "e.g. fighting_stance -> FightingStance)"
        ),
    )
    parser.add_argument(
        "--reference-dir",
        default="reference_poses",
        help="Reference pose base directory (default: reference_poses)",
    )
    parser.add_argument(
        "--weights",
        default="yolo26n-pose.pt",
        help="Pose model weights (default: yolo26n-pose.pt)",
    )
    parser.add_argument(
        "--target-technique",
        default="_capture_only",
        help="Target technique sent to action_recognition.py (default: _capture_only)",
    )
    parser.add_argument(
        "--num-video-sequence-samples",
        type=int,
        default=150,
        help="Number of samples in each saved reference sequence (default: 150)",
    )
    parser.add_argument(
        "--reference-search-max-frames",
        type=int,
        default=1800,
        help="Maximum frames scanned to find best window (default: 1800)",
    )
    parser.add_argument(
        "--ref-min-motion-energy",
        type=float,
        default=0.05,
        help="Minimum motion energy gate for reference extraction (default: 0.05)",
    )
    parser.add_argument(
        "--ref-min-return-closure",
        type=float,
        default=0.0,
        help="Minimum return closure gate for extraction (default: 0.0)",
    )
    parser.add_argument(
        "--ref-min-score-gate",
        type=float,
        default=0.0,
        help="Minimum score gate for extraction (default: 0.0)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without running them",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    project_root = PROJECT_ROOT

    golden_dir_name = args.golden_technique_dir.strip() or _to_pascal_case(args.technique_key)
    reference_root = _resolve_project_path(args.reference_dir)
    golden_dir = reference_root / "Golden_Seeds" / golden_dir_name
    out_dir = reference_root / args.technique_key
    out_dir.mkdir(parents=True, exist_ok=True)

    if not golden_dir.exists() or not golden_dir.is_dir():
        print(f"error: Golden Seeds folder not found: {golden_dir}")
        return 2

    candidate_files = sorted(
        [
            p
            for p in golden_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ],
        key=lambda p: p.name.lower(),
    )
    if not candidate_files:
        print(f"error: no video files found in {golden_dir}")
        return 2

    angle_counters: dict[str, int] = defaultdict(int)
    planned: list[tuple[Path, str, int, str, Path, list[str]]] = []
    skipped_unknown = 0
    skipped_exists = 0

    for file_path in candidate_files:
        angle = _infer_angle_from_filename(file_path.name)
        if not angle:
            print(f"skip (unknown angle): {file_path.name}")
            skipped_unknown += 1
            continue

        angle_counters[angle] += 1
        idx = angle_counters[angle]
        record_reference = f"{args.technique_key}__{angle}_{idx:02d}"
        out_file = out_dir / f"{angle}_{idx:02d}.npy"

        if out_file.exists() and not args.overwrite:
            print(f"skip (exists): {out_file}")
            skipped_exists += 1
            continue

        cmd = [
            sys.executable,
            "action_recognition.py",
            "--weights",
            args.weights,
            "--source",
            str(file_path),
            "--record-reference",
            record_reference,
            "--target-technique",
            args.target_technique,
            "--reference-dir",
            str(reference_root),
            "--reference-capture-mode",
            "best_window",
            "--num-video-sequence-samples",
            str(args.num_video_sequence_samples),
            "--skip-frame",
            "1",
            "--record-reference-max-saves",
            "1",
            "--reference-capture-cooldown-frames",
            "24",
            "--disable-video-classifier",
            "--no-display",
            "--auto-exit-after-reference",
            "--reference-search-max-frames",
            str(args.reference_search_max_frames),
            "--person-selection-mode",
            "most_motion",
            "--disable-structured-storage",
            "--ref-min-motion-energy",
            str(args.ref_min_motion_energy),
            "--ref-min-return-closure",
            str(args.ref_min_return_closure),
            "--ref-min-score-gate",
            str(args.ref_min_score_gate),
        ]
        planned.append((file_path, angle, idx, record_reference, out_file, cmd))

    if not planned:
        print("nothing to run")
        print(
            "summary:",
            {
                "planned": 0,
                "skipped_unknown": skipped_unknown,
                "skipped_exists": skipped_exists,
            },
        )
        return 0

    print(f"golden folder: {golden_dir}")
    print(f"output folder: {out_dir}")
    print(f"planned runs: {len(planned)}")

    if args.dry_run:
        for i, (file_path, angle, idx, record_reference, out_file, cmd) in enumerate(planned, start=1):
            print(f"[{i}/{len(planned)}] {file_path.name} -> {record_reference} -> {out_file.name}")
            print(" ", " ".join(cmd))
        print(
            "summary:",
            {
                "planned": len(planned),
                "skipped_unknown": skipped_unknown,
                "skipped_exists": skipped_exists,
                "dry_run": True,
            },
        )
        return 0

    succeeded = 0
    failed = 0
    for i, (file_path, angle, idx, record_reference, out_file, cmd) in enumerate(planned, start=1):
        print(f"[{i}/{len(planned)}] running {file_path.name} -> {record_reference}")
        result = subprocess.run(cmd, cwd=project_root)
        if result.returncode == 0 and out_file.exists():
            print(f"  saved: {out_file}")
            succeeded += 1
        else:
            print(f"  failed rc={result.returncode}: {record_reference}")
            failed += 1

    print(
        "summary:",
        {
            "planned": len(planned),
            "succeeded": succeeded,
            "failed": failed,
            "skipped_unknown": skipped_unknown,
            "skipped_exists": skipped_exists,
        },
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
