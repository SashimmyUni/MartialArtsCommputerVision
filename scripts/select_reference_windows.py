"""Stage 2 CLI: turn cached detection tables into reference ``.npy`` files.

Runs the window selection in ``reference_selection`` over whatever
``extract_tracks.py`` has already cached. No model is loaded, no video is
decoded and nothing is downloaded, so this is the command to re-run after
changing a capture gate — the part that used to require a full re-download and
re-inference of every source.

Usage::

    python scripts/select_reference_windows.py --dry-run
    python scripts/select_reference_windows.py --technique jab
    python scripts/select_reference_windows.py --examples-per-angle 4
    python scripts/select_reference_windows.py --ref-min-return-closure 0.30 --overwrite

Source diversity is preserved by default: each of a row's distinct videos
contributes its single best window before any video is asked for a second one.
The saving comes from each video being decoded once rather than once per
example. Raise ``--max-windows-per-video`` to trade diversity for speed on rows
that are short on sources.
"""

from __future__ import annotations

import argparse
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
        description="Select reference windows from cached pose detections (no GPU, no video, no network)."
    )
    parser.add_argument(
        "--plan",
        default="reference_poses/generated_capture_plan_all_labels.csv",
        help="capture plan CSV",
    )
    parser.add_argument("--technique", default=None, help="limit to one technique")
    parser.add_argument("--reference-dir", default="reference_poses", help="reference bank root")
    parser.add_argument(
        "--capture-seed-reference-dir",
        default=None,
        help="optional Golden Seed bank used only for capture gating",
    )
    parser.add_argument(
        "--examples-per-angle",
        type=int,
        default=4,
        help="target saved examples per technique/angle (default: 4)",
    )
    parser.add_argument(
        "--max-windows-per-video",
        type=int,
        default=1,
        help=(
            "windows one video may contribute before other sources are tried "
            "(default: 1, preserving today's source diversity; 0 = unlimited)"
        ),
    )
    parser.add_argument(
        "--min-gap-frames",
        type=int,
        default=24,
        help=(
            "minimum distance between the centres of two windows taken from the same "
            "video (default: 24, the live path's reference_capture_cooldown_frames)"
        ),
    )
    parser.add_argument(
        "--max-self-similarity",
        type=float,
        default=95.0,
        help=(
            "reject a window scoring at least this against one already accepted, so K "
            "windows from one video are not the same action repeated (0 disables)"
        ),
    )
    parser.add_argument(
        "--search-max-frames",
        type=int,
        default=1800,
        help="frames of each video to consider (default: 1800, matching capture; 0 = all cached)",
    )
    parser.add_argument(
        "--score-topk",
        type=int,
        default=0,
        help=(
            "cosine-prescreen the reference bank and only DTW-score the top K when ranking "
            "candidates (default: 0 = score the whole bank, matching the live capture path "
            "exactly). That DTW sweep is essentially all of this stage's cost, so a small K "
            "is much faster at the price of an approximate ranking"
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing examples")
    parser.add_argument("--dry-run", action="store_true", help="report picks without writing .npy files")

    gates = parser.add_argument_group("gate overrides (default: the technique's capture profile)")
    gates.add_argument("--num-video-sequence-samples", type=int, default=None)
    gates.add_argument("--ref-min-motion-energy", type=float, default=None)
    gates.add_argument("--ref-min-return-closure", type=float, default=None)
    gates.add_argument("--ref-min-score-gate", type=float, default=None)
    gates.add_argument("--reference-sequence-mode", default=None,
                       choices=["fixed", "event_centered", "stance_cycle"])
    return parser.parse_args()


def _existing_angle_examples(technique_dir: Path, angle: str) -> list[Path]:
    """Mirror of the batch runner's skip-if-satisfied check."""
    files: list[Path] = []
    base = technique_dir / f"{angle}.npy"
    if base.exists():
        files.append(base)
    files.extend(sorted(technique_dir.glob(f"{angle}_*.npy")))
    return files


def _profile_for(technique: str, args: argparse.Namespace) -> rs.SelectionConfig:
    """Per-technique capture profile, with any explicit CLI overrides applied.

    Delegates to run_reference_collection_batch's own lookup rather than keeping
    a second copy. That lookup also resolves karate techniques through the
    catalogue's ``capture_profile`` column, so a private copy here would quietly
    fall back to the generic profile for every technique in
    reference_poses/karate_techniques.csv.
    """
    from run_reference_collection_batch import _capture_profile_for_technique

    # The shared lookup reads these four off its own args namespace; supply the
    # batch runner's defaults for the ones this CLI does not expose, so an
    # unspecified value means "leave the profile alone" here too.
    profile_args = argparse.Namespace(
        num_video_sequence_samples=(
            args.num_video_sequence_samples if args.num_video_sequence_samples is not None else 20
        ),
        ref_min_return_closure=(
            args.ref_min_return_closure if args.ref_min_return_closure is not None else 0.20
        ),
        capture_seed_min_score=0.0,
        capture_seed_max_score=100.0,
    )
    profile = _capture_profile_for_technique(technique, profile_args)
    return rs.SelectionConfig.from_profile(
        profile,
        num_video_sequence_samples=args.num_video_sequence_samples,
        ref_min_motion_energy=args.ref_min_motion_energy,
        ref_min_return_closure=args.ref_min_return_closure,
        ref_min_score_gate=args.ref_min_score_gate,
        reference_sequence_mode=args.reference_sequence_mode,
        search_max_frames=args.search_max_frames,
        score_topk=args.score_topk,
    )


def main() -> int:
    args = parse_args()
    if args.examples_per_angle < 1:
        print("--examples-per-angle must be >= 1")
        return 2

    plan_path = Path(args.plan)
    if not plan_path.is_absolute():
        plan_path = PROJECT_ROOT / plan_path
    if not plan_path.exists():
        print(f"plan file not found: {plan_path}")
        return 2

    reference_root = Path(args.reference_dir)
    if not reference_root.is_absolute():
        reference_root = PROJECT_ROOT / reference_root

    rows = rc.plan_rows(plan_path)
    if args.technique:
        wanted = args.technique.strip().lower()
        rows = [r for r in rows if (r.get("technique") or "").strip().lower() == wanted]
    if not rows:
        print("no matching command_ready rows")
        return 2

    # Loaded once and mutated as examples are saved, so later rows are ranked
    # against earlier ones — what the old flow got by reloading from disk in
    # every subprocess.
    references = ar.load_reference_pose_library(str(reference_root))
    seed_references = {}
    if args.capture_seed_reference_dir:
        seed_dir = Path(args.capture_seed_reference_dir)
        if not seed_dir.is_absolute():
            seed_dir = PROJECT_ROOT / seed_dir
        seed_references = ar.load_reference_pose_library(str(seed_dir))
        print(f"capture seed bank: {len(seed_references)} technique(s) from {seed_dir}")

    print(f"plan: {plan_path.name}  rows: {len(rows)}  examples/angle: {args.examples_per_angle}")
    print(f"max windows per video: {args.max_windows_per_video or 'unlimited'}")
    if args.dry_run:
        print("dry run: no .npy files will be written")

    # Keyed by (video_id, profile) because the same video serves several rows and
    # replaying it is the slowest part of this stage.
    candidate_cache: dict[tuple[str, tuple], list[rs.Candidate]] = {}

    saved_total = 0
    rows_completed = 0
    rows_short = 0
    rows_skipped = 0
    missing_caches: set[str] = set()

    for i, row in enumerate(rows, start=1):
        technique = (row.get("technique") or "").strip()
        angle = (row.get("angle") or "").strip()
        if not technique or not angle:
            print(f"[{i}/{len(rows)}] skip malformed row (CSV line {row.get('_csv_line')})")
            continue

        technique_dir = reference_root / technique
        technique_dir.mkdir(parents=True, exist_ok=True)
        existing = _existing_angle_examples(technique_dir, angle)
        if args.overwrite and existing and not args.dry_run:
            for path in existing:
                path.unlink(missing_ok=True)
            existing = []
        elif args.overwrite and existing and args.dry_run:
            existing = []

        needed = args.examples_per_angle - len(existing)
        if needed <= 0:
            rows_skipped += 1
            continue

        source_urls = rc.collect_source_urls(row)
        if not source_urls:
            print(f"[{i}/{len(rows)}] {technique}/{angle}: no source URLs")
            rows_short += 1
            continue

        config = _profile_for(technique, args)
        config_key = tuple(sorted(vars(config).items()))
        reference_key_base = (row.get("record_reference_key") or row.get("reference_key") or "").strip()
        technique_key = ar._split_reference_key(reference_key_base)[0] if reference_key_base else technique

        print(f"[{i}/{len(rows)}] {technique}/{angle}: need {needed}, {len(source_urls)} source(s)")

        pools: dict[str, list[rs.Candidate]] = {}
        for url in source_urls:
            video_id = rc.video_id_for_source(url)
            if video_id in pools:
                continue
            pool = candidate_cache.get((video_id, config_key))
            if pool is None:
                cache = rc.load_best_track_cache(video_id, needed_frames=config.search_max_frames)
                if cache is None:
                    missing_caches.add(url)
                    pool = []
                else:
                    pool = rs.iter_candidates(cache, config)
                # Videos are shared across rows — 60 of the plan's URLs appear in
                # more than one — so the replay is memoised per (video, profile).
                candidate_cache[(video_id, config_key)] = pool
            pools[video_id] = pool

        picks = rs.select_across_sources(
            pools,
            needed,
            references=references,
            seed_references=seed_references,
            technique_key=technique_key,
            config=config,
            max_windows_per_video=args.max_windows_per_video,
            min_gap_frames=args.min_gap_frames,
            max_self_similarity=args.max_self_similarity,
        )

        if not picks:
            print("  - no window passed the gates in any cached source")
            rows_short += 1
            continue

        start_idx = len(existing) + 1
        for offset, (video_id, candidate) in enumerate(picks):
            ex_idx = start_idx + offset
            indexed_key = f"{technique}__{angle}_{ex_idx:02d}"
            out_path = technique_dir / f"{angle}_{ex_idx:02d}.npy"
            detail = (
                f"frames {candidate.start_frame}-{candidate.last_frame} of {video_id}, "
                f"{candidate.pose_seq.shape[0]} samples, selection {candidate.selection_score:.1f} "
                f"(motion {candidate.energy:.3f}, closure {candidate.closure:.3f})"
            )
            if args.dry_run:
                print(f"  - would save {out_path.name}: {detail}")
                continue

            saved_path = ar.save_reference_pose(
                str(reference_root), indexed_key, candidate.pose_seq, append_indexed=False
            )
            ar._put_reference(
                refs=references,
                technique=technique_key,
                angle=saved_path.stem,
                sequence=candidate.pose_seq,
            )
            ar._write_reference_meta(
                reference_path=saved_path,
                technique=indexed_key,
                source=str(candidate_source(row, video_id)),
                num_frames=int(candidate.pose_seq.shape[0]),
            )
            print(f"  - saved {saved_path.name}: {detail}")
            saved_total += 1

        if len(picks) >= needed:
            rows_completed += 1
        else:
            print(f"  - only {len(picks)}/{needed} example(s) found")
            rows_short += 1

    print(
        "summary:",
        {
            "rows": len(rows),
            "completed": rows_completed,
            "short": rows_short,
            "already_satisfied": rows_skipped,
            "examples_saved": saved_total,
            "dry_run": args.dry_run,
        },
    )
    if missing_caches:
        print(f"{len(missing_caches)} source(s) have no cached extraction. Run:")
        print("  python scripts/extract_tracks.py --from-plan")
    return 0 if rows_short == 0 else 1


def candidate_source(row: dict[str, str], video_id: str) -> str:
    """The plan URL a cached video came from, for the reference sidecar."""
    for url in rc.collect_source_urls(row):
        if rc.video_id_for_source(url) == video_id:
            return url
    return video_id


if __name__ == "__main__":
    raise SystemExit(main())
