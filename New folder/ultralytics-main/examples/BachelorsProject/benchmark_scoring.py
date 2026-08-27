"""Video-free scoring microbenchmark for the trainer's per-frame scoring core.

The dominant per-frame cost in ``action_recognition.py`` is
``_best_reference_match``: for the target technique, it DTW/cosine/angle-scores
the live pose window against every reference in that technique's bank. This
script times exactly that call, once per technique, using the committed
``keypoints/track_*.npy`` files as stand-ins for a live pose window — no video
or webcam needed, which matters here since the repo has no input videos.

Run from this directory (or anywhere — paths resolve relative to this file):

    python benchmark_scoring.py [--reps N] [--technique NAME ...]

Prints, per technique: reference count, sum-of-T^2 (the Python-level DTW cell
count the old nested-loop implementation would have iterated), and ms per
``_best_reference_match`` call, plus an overall weighted average. Also reports
the module-level reference-prep cache's hit rate so a second call's speed
isn't mistaken for the steady-state number — the FIRST call per technique
pays the one-time normalize+resample cost for that bank; everything after is
what a real run actually experiences frame to frame.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import action_recognition as ar  # noqa: E402


def _load_user_window(keypoints_dir: Path, num_video_sequence_samples: int = 8) -> np.ndarray:
    """A representative live pose window: the longest committed track,
    trimmed/padded the same way run() feeds ``_best_reference_match``
    (``stacked_seq[-num_video_sequence_samples:]``).
    """
    track_files = sorted(keypoints_dir.glob("track_*.npy"))
    if not track_files:
        raise SystemExit(f"no keypoint tracks found in {keypoints_dir}")

    best = None
    for fp in track_files:
        arr = np.load(fp)
        if arr.ndim == 3 and arr.shape[1] >= 10:
            if best is None or arr.shape[0] > best.shape[0]:
                best = arr.astype(np.float32)
    if best is None:
        raise SystemExit(f"no usable keypoint window found in {keypoints_dir}")
    return best[-num_video_sequence_samples:] if best.shape[0] >= num_video_sequence_samples else best


def benchmark_technique(
    technique: str, bank: dict[str, np.ndarray], user_seq: np.ndarray, reps: int
) -> dict[str, object]:
    sum_t2 = sum(seq.shape[0] ** 2 for seq in bank.values())

    # First call pays the one-time reference normalize+resample cost for this
    # bank (populates the module-level _REF_PREP_CACHE); report it separately
    # so it isn't mistaken for steady-state per-frame cost.
    t0 = time.perf_counter()
    ar._best_reference_match(user_seq, bank, technique)
    first_call_ms = (time.perf_counter() - t0) * 1000.0

    timings_ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        ar._best_reference_match(user_seq, bank, technique)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "technique": technique,
        "num_refs": len(bank),
        "sum_t2": sum_t2,
        "first_call_ms": first_call_ms,
        "mean_ms": statistics.mean(timings_ms),
        "median_ms": statistics.median(timings_ms),
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", default=str(SCRIPT_DIR / "reference_poses"))
    parser.add_argument("--keypoints-dir", default=str(SCRIPT_DIR / "keypoints"))
    parser.add_argument("--reps", type=int, default=30, help="timed calls per technique after warmup (default: 30)")
    parser.add_argument(
        "--technique", action="append", default=None, help="limit to specific technique(s); repeatable"
    )
    parser.add_argument("--score-topk", type=int, default=0, help="also benchmark with --score-topk N prescreen")
    args = parser.parse_args()

    references = ar.load_reference_pose_library(args.reference_dir)
    if not references:
        raise SystemExit(f"no references loaded from {args.reference_dir}")

    user_seq = _load_user_window(Path(args.keypoints_dir))
    techniques = args.technique or sorted(references.keys())

    print(f"user window shape: {user_seq.shape}, reps per technique: {args.reps}\n")
    header = f"{'technique':<18} {'refs':>5} {'sum(T^2)':>10} {'first_ms':>10} {'mean_ms':>9} {'median_ms':>10}"
    print(header)
    print("-" * len(header))

    results = []
    for technique in techniques:
        if technique not in references:
            print(f"  (skip: '{technique}' not in reference library)")
            continue
        result = benchmark_technique(technique, references[technique], user_seq, args.reps)
        results.append(result)
        print(
            f"{result['technique']:<18} {result['num_refs']:>5} {result['sum_t2']:>10} "
            f"{result['first_call_ms']:>10.3f} {result['mean_ms']:>9.3f} {result['median_ms']:>10.3f}"
        )

    if results:
        total_calls_ms = sum(r["mean_ms"] for r in results)
        print(f"\nsum of per-technique mean ms/call: {total_calls_ms:.3f}")
        print(f"overall mean ms/call: {statistics.mean(r['mean_ms'] for r in results):.3f}")

    if args.score_topk > 0:
        print(f"\n--score-topk {args.score_topk} comparison:")
        print(header)
        print("-" * len(header))
        for technique in techniques:
            if technique not in references:
                continue
            bank = references[technique]
            if len(bank) <= args.score_topk:
                continue
            timings_ms = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                ar._best_reference_match(user_seq, bank, technique, topk=args.score_topk)
                timings_ms.append((time.perf_counter() - t0) * 1000.0)
            print(
                f"{technique:<18} {len(bank):>5} {sum(s.shape[0] ** 2 for s in bank.values()):>10} "
                f"{'':>10} {statistics.mean(timings_ms):>9.3f} {statistics.median(timings_ms):>10.3f}"
            )


if __name__ == "__main__":
    main()
