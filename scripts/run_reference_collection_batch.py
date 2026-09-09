from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ultralytics.utils.tqdm import TQDM

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import reference_cache as rc  # noqa: E402  (needs SCRIPT_DIR on sys.path)


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
        default=0.0,
        help=(
            "sleep time between capture jobs, e.g. to cool a thermally-limited CPU (default: 0.0 — "
            "was 8.0; with --jobs running captures concurrently there is little reason to also "
            "throttle with a fixed sleep, but pass a positive value if your machine needs it)"
        ),
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=6,
        help="max CPU threads for each child capture process's BLAS/OMP pool (default: 6)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help=(
            "number of plan rows (technique/angle) to capture concurrently, each spawning its own "
            "action_recognition.py subprocess with its own YOLO model on the GPU (default: 2). "
            "Each subprocess holds its own CUDA context + model in VRAM — raise this only if you've "
            "confirmed your GPU has headroom for it; pass --jobs 1 to fall back to the original "
            "fully sequential behaviour"
        ),
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
        "--legacy",
        action="store_true",
        help=(
            "use the original flow: one action_recognition.py subprocess per saved example, "
            "re-decoding and re-inferring a video for each one. Kept for comparison — the "
            "staged pipeline should produce the same references far more cheaply"
        ),
    )
    parser.add_argument(
        "--max-windows-per-video",
        type=int,
        default=1,
        help=(
            "staged pipeline: windows one video may contribute to a row before other sources "
            "are tried (default: 1, keeping one example per video as before; 0 = unlimited)"
        ),
    )
    parser.add_argument(
        "--score-topk",
        type=int,
        default=0,
        help=(
            "staged pipeline: cosine-prescreen the reference bank and only DTW the top K when "
            "ranking candidates (default: 0 = exact, matching the legacy path)"
        ),
    )
    parser.add_argument(
        "--no-video-cache",
        dest="video_cache",
        action="store_false",
        help=(
            "stream sources straight from YouTube instead of downloading them once into "
            "cache/videos/ first. Streaming is what made long batches fragile (a dropped "
            "connection kills the run) and re-fetched videos shared across plan rows"
        ),
    )
    parser.add_argument(
        "--prefetch-only",
        action="store_true",
        help="download every source the plan names into cache/videos/, then exit without capturing",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help=(
            "run the pose model in half precision on CUDA. Faster, but keypoints can shift in "
            "the last decimals, so captured windows may differ slightly from an FP32 run "
            "(default: off, matching prior behaviour)"
        ),
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

# URL parsing lives in reference_cache so prefetch_sources.py and the extraction
# stage read the plan the same way this does; a second copy would drift.
_collect_source_urls = rc.collect_source_urls


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


def _run_stage(name: str, cmd: list[str], project_root: Path, env: dict) -> bool:
    print(f"\n=== {name} ===")
    print("  " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=project_root, env=env)
    if result.returncode != 0:
        print(f"{name} exited {result.returncode}")
        return False
    return True


def _run_staged_pipeline(
    args: argparse.Namespace,
    ready_rows: list[dict[str, str]],
    project_root: Path,
    child_env: dict,
) -> int:
    """prefetch -> extract once per video -> select windows from the cache.

    Replaces one subprocess per saved example (208 for the ready plan, each
    paying model load and CUDA setup before decoding a frame) with one
    extraction pass per distinct video and a pure-numpy selection pass. Videos
    shared across rows are decoded once rather than once per row.

    Run as subprocesses rather than imports so the GPU stage's memory is
    released before selection starts, and so either stage can be re-run on its
    own — re-tuning a gate only needs the third one.
    """
    if args.video_cache:
        rc_result = _prefetch_plan_sources(ready_rows)
        if rc_result != 0:
            return rc_result

    extract_cmd = [
        sys.executable,
        str(Path("scripts") / "extract_tracks.py"),
        "--from-plan",
        "--weights", "yolo26n-pose.pt",
        "--max-frames", "1800",
    ]
    if args.fp16:
        extract_cmd.append("--fp16")
    if not args.video_cache:
        extract_cmd.append("--no-video-cache")
    if not _run_stage("extract pose detections (once per video)", extract_cmd, project_root, child_env):
        return 1

    select_cmd = [
        sys.executable,
        str(Path("scripts") / "select_reference_windows.py"),
        "--examples-per-angle", str(args.examples_per_angle),
        "--max-windows-per-video", str(args.max_windows_per_video),
        "--score-topk", str(args.score_topk),
    ]
    if args.overwrite:
        select_cmd.append("--overwrite")
    if args.capture_seed_reference_dir:
        select_cmd.extend(["--capture-seed-reference-dir", args.capture_seed_reference_dir])
    if args.num_video_sequence_samples != 20:
        select_cmd.extend(["--num-video-sequence-samples", str(args.num_video_sequence_samples)])
    if abs(float(args.ref_min_return_closure) - 0.20) > 1e-9:
        select_cmd.extend(["--ref-min-return-closure", str(args.ref_min_return_closure)])
    if not _run_stage("select reference windows from cache", select_cmd, project_root, child_env):
        return 1

    print(
        "\nDone. To re-tune a capture gate, re-run only the last stage — it needs no "
        "GPU, video or network:\n"
        "  python scripts/select_reference_windows.py --ref-min-return-closure 0.30 --overwrite"
    )
    return 0


def _prefetch_plan_sources(ready_rows: list[dict[str, str]]) -> int:
    """Download every distinct source the ready rows name, then stop.

    Separated from capture so the slow, failure-prone part of a batch can be
    done once up front and retried on its own, instead of dying halfway
    through an overnight capture run.
    """
    by_id: dict[str, str] = {}
    for row in ready_rows:
        for url in _collect_source_urls(row):
            by_id.setdefault(rc.video_id_for_source(url), url)
    missing = [(vid, url) for vid, url in by_id.items() if rc.find_cached_video(vid) is None]
    print(f"prefetch: {len(by_id)} distinct source(s), {len(missing)} to download")
    failed = 0
    for i, (vid, url) in enumerate(missing, start=1):
        print(f"  [{i}/{len(missing)}] {vid} {url}")
        if rc.ensure_local_video(url) is None:
            failed += 1
    print("prefetch summary:", {"distinct": len(by_id), "downloaded": len(missing) - failed, "failed": failed})
    return 0


def _resolve_source(url: str, use_cache: bool) -> str:
    """Prefer a locally cached copy of ``url``; fall back to the URL itself.

    Falling back rather than failing is deliberate: a missing or broken
    yt-dlp should make capture slower, not impossible.
    """
    if not use_cache or not rc.is_url(url):
        return url
    local = rc.ensure_local_video(url, quiet=True)
    return str(local) if local is not None else url


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
    if args.jobs < 1:
        print("jobs must be >= 1")
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

    if args.prefetch_only:
        return _prefetch_plan_sources(ready_rows)

    if preflight_issues and not args.allow_source_reuse:
        print("aborting batch due to preflight failures. Add more URLs or use --allow-source-reuse")
        return 2

    if not args.legacy:
        return _run_staged_pipeline(args, ready_rows, project_root, child_env)

    print(
        "--legacy: one capture subprocess per saved example, re-decoding a video for each. "
        "The default staged pipeline decodes each video once."
    )

    def _process_row(i: int, row: dict[str, str]) -> dict[str, object]:
        """Run one plan row (technique/angle) to completion: skip/overwrite checks,
        then one action_recognition.py subprocess per needed example (retrying
        across that row's source URLs), fully self-contained so it's safe to run
        concurrently with other rows — it only ever touches files named for its
        own (technique, angle), never another row's.
        """
        technique = (row.get("technique") or "").strip()
        angle = (row.get("angle") or "").strip()
        source_urls = _collect_source_urls(row)
        reference_key = (row.get("record_reference_key") or row.get("reference_key") or "").strip()
        if not technique or not angle or not source_urls or not reference_key:
            print(f"[{i}/{len(ready_rows)}] skip malformed row")
            return {"outcome": "failed", "saved": 0}

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
            return {"outcome": "skip_existing", "saved": 0}

        needed = max(0, args.examples_per_angle - len(existing_files))
        if needed > len(source_urls) and not args.allow_source_reuse:
            print(
                f"[{i}/{len(ready_rows)}] failed: {technique}/{angle} needs {needed} distinct source URLs, "
                f"but only {len(source_urls)} provided"
            )
            return {"outcome": "failed", "saved": 0}

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
                local_source = _resolve_source(source_url, args.video_cache)
                cmd = [
                    sys.executable,
                    "action_recognition.py",
                    "--weights",
                    "yolo26n-pose.pt",
                    "--source",
                    local_source,
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
                    # NOT "keypoints": that directory is committed and is the pinned
                    # fixture for test_scoring_equivalence.py and benchmark_scoring.py.
                    # Capture runs used to overwrite it on every one of the ~208 jobs
                    # (and, under --jobs, two at once), silently invalidating the only
                    # guard the scoring core has. Per-key subdirectory under gitignored
                    # data/ so concurrent jobs cannot collide either.
                    "--save-kpts-dir",
                    str(Path("data") / "capture_keypoints" / indexed_key),
                    "--record-reference-max-saves",
                    "1",
                    "--reference-capture-cooldown-frames",
                    "24",
                    "--disable-video-classifier",
                    "--no-display",
                    # Everything below this line only removes work whose result is
                    # discarded during capture: the default --output-path writes a full
                    # annotated mp4 nobody reads (and concurrent --jobs workers raced on
                    # the same file), --no-display gates only cv2.imshow so boxes/overlay
                    # were still drawn every frame, and --capture-only skips the live
                    # trainer score that is computed and thrown away. Capture itself is
                    # untouched: --capture-only is not --disable-trainer, which would
                    # disable capture too.
                    "--output-path",
                    "",
                    "--no-boxes",
                    "--no-overlay-pose",
                    "--capture-only",
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
                if args.fp16:
                    cmd.append("--fp16")
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
                if local_source != source_url:
                    print(f"      using cached copy: {local_source}")
                try:
                    result = subprocess.run(cmd, cwd=project_root, timeout=1200, env=child_env)
                except subprocess.TimeoutExpired:
                    print(f"    timed out after 1200s (20 minutes) on source {attempt_idx}: {source_url}")
                    continue

                if result.returncode == 0 and out_file.exists():
                    saved_for_row += 1
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
            return {"outcome": "failed", "saved": saved_for_row}
        print(f"[{i}/{len(ready_rows)}] saved {saved_for_row} new example(s) for {technique}/{angle}")
        return {"outcome": "completed", "saved": saved_for_row}

    completed = 0
    skipped_existing = 0
    failed = 0
    examples_saved = 0
    counters_lock = threading.Lock()
    batch_progress = TQDM(total=len(ready_rows), desc="capture batch", unit="job")

    def _record_result(result: dict[str, object]) -> None:
        nonlocal completed, skipped_existing, failed, examples_saved
        with counters_lock:
            outcome = result["outcome"]
            examples_saved += int(result["saved"])
            if outcome == "completed":
                completed += 1
            elif outcome == "skip_existing":
                skipped_existing += 1
            else:
                failed += 1
            batch_progress.update(1)
            batch_progress.set_postfix(completed=completed, skipped=skipped_existing, failed=failed)

    if args.jobs <= 1:
        # Exact original sequential path — no thread pool involved.
        for i, row in enumerate(ready_rows, start=1):
            _record_result(_process_row(i, row))
    else:
        print(f"running up to {args.jobs} capture job(s) concurrently")
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(_process_row, i, row) for i, row in enumerate(ready_rows, start=1)]
            for future in as_completed(futures):
                _record_result(future.result())

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
