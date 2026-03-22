from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from ultralytics.utils.tqdm import TQDM


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _resolve_project_path(path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else PROJECT_ROOT / p


PUNCH_PROFILE = {
    "reference_sequence_mode": "stance_cycle",
    "num_video_sequence_samples": 20,
    "ref_min_motion_energy": 0.03,
    "ref_min_return_closure": 0.22,
    "capture_seed_min_score": 74.0,
    "capture_seed_max_score": 93.0,
    "ref_stance_start_threshold": 0.17,
    "ref_stance_end_threshold": 0.12,
    "ref_stance_peak_threshold": 0.30,
    "ref_stance_min_frames": 20,
    "ref_stance_hold_frames": 4,
}

KICK_PROFILE = {
    "reference_sequence_mode": "stance_cycle",
    "num_video_sequence_samples": 24,
    "ref_min_motion_energy": 0.05,
    "ref_min_return_closure": 0.28,
    "capture_seed_min_score": 70.0,
    "capture_seed_max_score": 91.0,
    "ref_stance_start_threshold": 0.20,
    "ref_stance_end_threshold": 0.14,
    "ref_stance_peak_threshold": 0.38,
    "ref_stance_min_frames": 24,
    "ref_stance_hold_frames": 4,
}

STANCE_PROFILE = {
    "reference_sequence_mode": "stance_cycle",
    "num_video_sequence_samples": 24,
    "ref_min_motion_energy": 0.015,
    "ref_min_return_closure": 0.10,
    "capture_seed_min_score": 78.0,
    "capture_seed_max_score": 96.0,
    "ref_stance_start_threshold": 0.12,
    "ref_stance_end_threshold": 0.08,
    "ref_stance_peak_threshold": 0.18,
    "ref_stance_min_frames": 24,
    "ref_stance_hold_frames": 5,
}

TECHNIQUE_CAPTURE_PROFILES = {
    "fighting_stance": STANCE_PROFILE,
    "jab": PUNCH_PROFILE,
    "cross": PUNCH_PROFILE,
    "hook": PUNCH_PROFILE,
    "uppercut": PUNCH_PROFILE,
    "elbow_strike": PUNCH_PROFILE,
    "front_kick": KICK_PROFILE,
    "side_kick": KICK_PROFILE,
    "roundhouse_kick": KICK_PROFILE,
    "back_kick": KICK_PROFILE,
    "spinning_back_kick": KICK_PROFILE,
    "knee_strike": KICK_PROFILE,
    "axe_kick": KICK_PROFILE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch reference capture commands from generated plan CSV.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-capture even if the target reference .npy already exists",
    )
    parser.add_argument(
        "--num-video-sequence-samples",
        type=int,
        default=20,
        help="window length to save per reference (default: 20)",
    )
    parser.add_argument(
        "--ref-min-return-closure",
        type=float,
        default=0.20,
        help="minimum return-closure required before accepting a reference (default: 0.20)",
    )
    parser.add_argument(
        "--examples-per-angle",
        type=int,
        default=4,
        help="target number of saved reference examples per technique/angle (default: 4)",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=8.0,
        help="sleep time between capture jobs to cool CPU (default: 8.0)",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=6,
        help="max CPU threads for child capture process (default: 6)",
    )
    parser.add_argument(
        "--capture-seed-reference-dir",
        type=str,
        default=None,
        help=(
            "optional reference directory used only for capture gating. Point this to a Golden Seed-derived "
            "reference bank when new captures should be similar but not identical"
        ),
    )
    parser.add_argument(
        "--capture-seed-min-score",
        type=float,
        default=0.0,
        help="minimum similarity score required against the capture seed bank (default: 0.0)",
    )
    parser.add_argument(
        "--capture-seed-max-score",
        type=float,
        default=100.0,
        help="maximum similarity score allowed against the capture seed bank (default: 100.0)",
    )
    parser.add_argument(
        "--allow-source-reuse",
        action="store_true",
        help="allow reusing source URLs when fewer distinct URLs are available than needed examples",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run CSV source-diversity validation and exit without starting capture",
    )
    return parser.parse_args()


def _existing_angle_examples(technique_dir: Path, angle: str) -> list[Path]:
    files: list[Path] = []
    base = technique_dir / f"{angle}.npy"
    if base.exists():
        files.append(base)
    files.extend(sorted(technique_dir.glob(f"{angle}_*.npy")))
    return files

def _parse_source_urls(raw_source: str) -> list[str]:
    """Parse one or more candidate source URLs from CSV field.

    Supports separators: newline, comma, semicolon, and pipe.
    """
    if not raw_source:
        return []
    parts = [p.strip() for p in re.split(r"[\n,;|]+", raw_source) if p.strip()]
    # preserve order while de-duplicating
    seen: set[str] = set()
    unique: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _collect_source_urls(row: dict[str, str]) -> list[str]:
    """Collect distinct source URLs from explicit source_url_1..source_url_4 columns and legacy source_url."""
    urls: list[str] = []

    for i in range(1, 5):
        cell = (row.get(f"source_url_{i}") or "").strip()
        if cell:
            urls.extend(_parse_source_urls(cell))

    # Backward compatibility with legacy single-column source_url and delimiter-packed values.
    legacy = (row.get("source_url") or "").strip()
    if legacy:
        urls.extend(_parse_source_urls(legacy))

    seen: set[str] = set()
    unique: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _preflight_distinct_sources(ready_rows: list[dict[str, str]], required_count: int) -> list[dict[str, str | int]]:
    issues: list[dict[str, str | int]] = []
    for row in ready_rows:
        technique = (row.get("technique") or "").strip()
        angle = (row.get("angle") or "").strip()
        csv_line = row.get("_csv_line", "?")
        source_urls = _collect_source_urls(row)
        distinct_count = len(source_urls)
        if distinct_count < required_count:
            issues.append(
                {
                    "csv_line": csv_line,
                    "technique": technique,
                    "angle": angle,
                    "distinct_sources": distinct_count,
                    "required_sources": required_count,
                }
            )
    return issues


def _normalize_key(text: str) -> str:
    return "_".join((text or "").strip().lower().replace("-", " ").split())


def _capture_profile_for_technique(technique: str, args: argparse.Namespace) -> dict[str, float | int | str]:
    profile = dict(TECHNIQUE_CAPTURE_PROFILES.get(_normalize_key(technique), {}))
    if not profile:
        profile = {
            "reference_sequence_mode": "fixed",
            "num_video_sequence_samples": args.num_video_sequence_samples,
            "ref_min_motion_energy": 0.02,
            "ref_min_return_closure": float(args.ref_min_return_closure),
            "capture_seed_min_score": float(args.capture_seed_min_score),
            "capture_seed_max_score": float(args.capture_seed_max_score),
            "ref_stance_start_threshold": 0.18,
            "ref_stance_end_threshold": 0.12,
            "ref_stance_peak_threshold": 0.30,
            "ref_stance_min_frames": 24,
            "ref_stance_hold_frames": 4,
        }

    if args.num_video_sequence_samples != 20:
        profile["num_video_sequence_samples"] = args.num_video_sequence_samples
    if abs(float(args.ref_min_return_closure) - 0.20) > 1e-9:
        profile["ref_min_return_closure"] = float(args.ref_min_return_closure)
    if abs(float(args.capture_seed_min_score) - 0.0) > 1e-9:
        profile["capture_seed_min_score"] = float(args.capture_seed_min_score)
    if abs(float(args.capture_seed_max_score) - 100.0) > 1e-9:
        profile["capture_seed_max_score"] = float(args.capture_seed_max_score)

    return profile


def main() -> int:
    args = parse_args()
    if args.examples_per_angle < 1:
        print("examples-per-angle must be >= 1")
        return 2
    if args.cooldown_seconds < 0:
        print("cooldown-seconds must be >= 0")
        return 2
    if args.cpu_threads < 1:
        print("cpu-threads must be >= 1")
        return 2
    if args.capture_seed_min_score < 0.0 or args.capture_seed_max_score > 100.0:
        print("capture-seed min/max scores must be in [0, 100]")
        return 2
    if args.capture_seed_min_score > args.capture_seed_max_score:
        print("capture-seed-min-score must be <= capture-seed-max-score")
        return 2

    project_root = PROJECT_ROOT
    plan_path = project_root / "reference_poses" / "generated_capture_plan_all_labels.csv"
    if not plan_path.exists():
        print(f"plan file not found: {plan_path}")
        return 2

    rows = list(csv.DictReader(plan_path.open(encoding="utf-8-sig")))
    ready_rows: list[dict[str, str]] = []
    for csv_line, row in enumerate(rows, start=2):  # header row is line 1
        if (row.get("command_ready", "").strip().lower() == "yes"):
            row_with_meta = dict(row)
            row_with_meta["_csv_line"] = str(csv_line)
            ready_rows.append(row_with_meta)

    print(f"loaded {len(rows)} total rows, {len(ready_rows)} ready row(s)")
    print(
        "throttle config:",
        {
            "cpu_threads": args.cpu_threads,
            "skip_frame": 1,
            "reference_search_max_frames": 1800,
            "cooldown_seconds": args.cooldown_seconds,
            "allow_source_reuse": args.allow_source_reuse,
            "capture_seed_reference_dir": args.capture_seed_reference_dir,
            "capture_seed_min_score": args.capture_seed_min_score,
            "capture_seed_max_score": args.capture_seed_max_score,
        },
    )

    capture_seed_reference_dir = None
    if args.capture_seed_reference_dir:
        capture_seed_reference_dir = str(_resolve_project_path(args.capture_seed_reference_dir))

    child_env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        child_env[var] = str(args.cpu_threads)

    required_sources = args.examples_per_angle
    preflight_issues = _preflight_distinct_sources(ready_rows, required_sources)
    if preflight_issues:
        print(
            f"preflight: {len(preflight_issues)} row(s) do not have at least "
            f"{required_sources} distinct source URL(s):"
        )
        for issue in preflight_issues:
            print(
                "  - "
                f"CSV line {issue['csv_line']}: "
                f"{issue['technique']}/{issue['angle']} "
                f"has {issue['distinct_sources']} distinct URL(s), "
                f"needs {issue['required_sources']}"
            )
    else:
        print(f"preflight: all ready rows have at least {required_sources} distinct source URL(s)")

    if args.preflight_only:
        return 0 if not preflight_issues else 2

    if preflight_issues and not args.allow_source_reuse:
        print("aborting batch due to preflight failures. Add more URLs or use --allow-source-reuse")
        return 2

    completed = 0
    skipped_existing = 0
    failed = 0
    examples_saved = 0
    batch_progress = TQDM(total=len(ready_rows), desc="capture batch", unit="job")

    for i, row in enumerate(ready_rows, start=1):
        technique = (row.get("technique") or "").strip()
        angle = (row.get("angle") or "").strip()
        source_urls = _collect_source_urls(row)
        reference_key = (row.get("record_reference_key") or row.get("reference_key") or "").strip()
        if not technique or not angle or not source_urls or not reference_key:
            print(f"[{i}/{len(ready_rows)}] skip malformed row")
            failed += 1
            batch_progress.update(1)
            batch_progress.set_postfix(completed=completed, skipped=skipped_existing, failed=failed)
            continue

        technique_dir = project_root / "reference_poses" / technique
        technique_dir.mkdir(parents=True, exist_ok=True)
        existing_files = _existing_angle_examples(technique_dir, angle)

        if args.overwrite and existing_files:
            for fp in existing_files:
                fp.unlink(missing_ok=True)
            existing_files = []

        if len(existing_files) >= args.examples_per_angle and not args.overwrite:
            print(
                f"[{i}/{len(ready_rows)}] skip existing: {technique}/{angle} "
                f"has {len(existing_files)} example(s)"
            )
            skipped_existing += 1
            batch_progress.update(1)
            batch_progress.set_postfix(completed=completed, skipped=skipped_existing, failed=failed)
            continue

        needed = max(0, args.examples_per_angle - len(existing_files))
        if needed > len(source_urls) and not args.allow_source_reuse:
            print(
                f"[{i}/{len(ready_rows)}] failed: {technique}/{angle} needs {needed} distinct source URLs, "
                f"but only {len(source_urls)} provided"
            )
            failed += 1
            batch_progress.update(1)
            batch_progress.set_postfix(completed=completed, skipped=skipped_existing, failed=failed)
            continue

        saved_for_row = 0
        row_failed = False
        print(
            f"[{i}/{len(ready_rows)}] running: {technique}/{angle} "
            f"need {needed} more example(s), sources available={len(source_urls)}"
        )
        profile = _capture_profile_for_technique(technique, args)
        print(f"  profile: {profile}")

        start_idx = len(existing_files) + 1
        for ex_idx in range(start_idx, start_idx + needed):
            indexed_key = f"{technique}__{angle}_{ex_idx:02d}"
            out_file = technique_dir / f"{angle}_{ex_idx:02d}.npy"
            source_idx = (ex_idx - start_idx) % len(source_urls)
            rotated_sources = source_urls[source_idx:] + source_urls[:source_idx]
            print(f"  - example {ex_idx:02d}: {indexed_key} (starting source {source_idx + 1}/{len(source_urls)})")
            example_saved = False
            for attempt_idx, source_url in enumerate(rotated_sources, start=1):
                cmd = [
                    sys.executable,
                    "action_recognition.py",
                    "--weights",
                    "yolo26n-pose.pt",
                    "--source",
                    source_url,
                    "--record-reference",
                    indexed_key,
                    "--reference-capture-mode",
                    "best_window",
                    "--reference-sequence-mode",
                    str(profile["reference_sequence_mode"]),
                    "--target-technique",
                    technique,
                    "--reference-dir",
                    "reference_poses",
                    "--num-video-sequence-samples",
                    str(profile["num_video_sequence_samples"]),
                    "--skip-frame",
                    "1",
                    "--save-kpts-dir",
                    "keypoints",
                    "--record-reference-max-saves",
                    "1",
                    "--reference-capture-cooldown-frames",
                    "24",
                    "--disable-video-classifier",
                    "--no-display",
                    "--auto-exit-after-reference",
                    "--reference-search-max-frames",
                    "1800",
                    "--person-selection-mode",
                    "most_motion",
                    "--disable-structured-storage",
                    "--ref-min-motion-energy",
                    f"{float(profile['ref_min_motion_energy']):.2f}",
                    "--ref-min-return-closure",
                    f"{float(profile['ref_min_return_closure']):.2f}",
                    "--ref-min-score-gate",
                    "0",
                    "--ref-stance-start-threshold",
                    f"{float(profile['ref_stance_start_threshold']):.2f}",
                    "--ref-stance-end-threshold",
                    f"{float(profile['ref_stance_end_threshold']):.2f}",
                    "--ref-stance-peak-threshold",
                    f"{float(profile['ref_stance_peak_threshold']):.2f}",
                    "--ref-stance-min-frames",
                    str(int(profile['ref_stance_min_frames'])),
                    "--ref-stance-hold-frames",
                    str(int(profile['ref_stance_hold_frames'])),
                ]
                if capture_seed_reference_dir:
                    cmd.extend(
                        [
                            "--capture-seed-reference-dir",
                            capture_seed_reference_dir,
                            "--capture-seed-min-score",
                            f"{float(profile['capture_seed_min_score']):.2f}",
                            "--capture-seed-max-score",
                            f"{float(profile['capture_seed_max_score']):.2f}",
                        ]
                    )

                print(f"    attempt {attempt_idx}/{len(rotated_sources)} source={source_url}")
                try:
                    result = subprocess.run(cmd, cwd=project_root, timeout=1200, env=child_env)
                except subprocess.TimeoutExpired:
                    print(f"    timed out after 1200s (20 minutes) on source {attempt_idx}: {source_url}")
                    continue

                if result.returncode == 0 and out_file.exists():
                    saved_for_row += 1
                    examples_saved += 1
                    example_saved = True
                    print(f"  - saved: {out_file}")
                    break

                print(f"    failed rc={result.returncode} on source {attempt_idx}: {source_url}")

            if not example_saved:
                print(f"  - all sources failed for {indexed_key}")
                row_failed = True
                break

            if args.cooldown_seconds > 0:
                print(f"  - cooldown: sleeping {args.cooldown_seconds:.1f}s")
                time.sleep(args.cooldown_seconds)

        if row_failed:
            failed += 1
        else:
            completed += 1
            print(f"[{i}/{len(ready_rows)}] saved {saved_for_row} new example(s) for {technique}/{angle}")

        batch_progress.update(1)
        batch_progress.set_postfix(completed=completed, skipped=skipped_existing, failed=failed)

    batch_progress.close()

    print(
        "summary:",
        {
            "completed": completed,
            "skipped_existing": skipped_existing,
            "failed": failed,
            "examples_saved": examples_saved,
            "examples_per_angle": args.examples_per_angle,
            "ready_total": len(ready_rows),
        },
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
