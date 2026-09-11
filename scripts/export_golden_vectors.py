"""Export golden test vectors pinning the scoring core, for verifying a port.

``test_scoring_equivalence.py`` guards the optimized Python scoring core against
a pinned copy of the original. This script does the same job across a language
boundary: it records what every layer of the scoring core produces for a fixed
set of inputs, so a Swift or TypeScript reimplementation can be checked against
the numbers this repo actually produces rather than against a reading of the
Python.

The vectors are **layered**, which is the whole point. A single end-to-end score
mismatch tells a porter nothing about where the port went wrong; six levels tell
them exactly which function diverged:

1. ``normalize``     -- ``normalize_pose_sequence`` per input window
2. ``resample``      -- ``resample_pose_sequence`` at several target lengths
3. ``dtw``           -- full ``_pairwise_frame_cost_matrix`` plus ``_dtw_min_plus``
4. ``compare``       -- ``compare_pose_sequence`` metric bundle, every angle
5. ``best_match``    -- ``_best_reference_match`` chosen angle and metrics
6. ``feedback``      -- ``generate_feedback`` strings

Inputs are the committed ``keypoints/track_*.npy`` fixtures, the same ones
``benchmark_scoring.py`` uses. They are embedded in the output as plain nested
lists so a consumer needs no ``.npy`` reader -- the golden file is self-contained
and is the only artifact a port needs.

**NaN is written as JSON ``null``.** ``normalize_pose_frame`` deliberately sets
sub-confidence joints to NaN, so level 1 output genuinely contains them and the
nulls are signal, not missing data. Bare ``NaN`` is not valid JSON and strict
parsers reject it.

Note where the NaNs stop, because it is easy to get backwards:
``resample_pose_sequence`` forward-fills, back-fills and interpolates every NaN
away (and writes 0.0 for a joint invalid in every frame), so levels 2 onward are
NaN-free and the "valid in both frames" masking inside
``_pairwise_frame_cost_matrix`` never actually fires in this pipeline. What a port
must reproduce exactly is therefore the *fill* behaviour, not the cost-matrix
masking -- the fill is what decides which numbers reach the comparison.

Run from anywhere; paths resolve against the repo root:

    python scripts/export_golden_vectors.py
    python scripts/export_golden_vectors.py --max-fixtures 8 --technique jab
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# action_recognition.py lives at the repo root, one level above this script.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from action_recognition import (  # noqa: E402
    _best_reference_match,
    _dtw_min_plus,
    _mirror_sequence,
    _pairwise_frame_cost_matrix,
    _prepare_reference_sequence,
    _technique_angle_category,
    compare_pose_sequence,
    dtw_pose_distance,
    generate_feedback,
    load_reference_pose_library,
    normalize_pose_sequence,
    resample_pose_sequence,
)

#: Bump when the layout changes in a way a consumer must notice.
GOLDEN_FORMAT_VERSION = 1

#: Default confidence gate, matching the scoring core.
CONF_THRESH = 0.2

#: Rolling window the trainer feeds the scorer: ``stacked_seq[-8:]``
#: (action_recognition.py:2803).
NUM_VIDEO_SEQUENCE_SAMPLES = 8

#: Target lengths exercised by the resample level. 1 and 4 hit the degenerate
#: branches in ``_resample_1d`` / ``resample_pose_sequence``; the rest are
#: representative reference lengths.
RESAMPLE_TARGET_LENGTHS = (1, 4, 8, 17, 33, 150)

#: Suggested comparison tolerances for a port. Coordinates are float32 through
#: the whole pipeline, so exact equality is not a reasonable bar; these are
#: tight enough to catch a real algorithmic divergence.
COMPARISON_TOLERANCES = {
    "normalized_coordinates_atol": 1e-5,
    "resampled_coordinates_atol": 1e-5,
    "cost_matrix_atol": 1e-5,
    "dtw_distance_atol": 1e-5,
    "score_atol": 1e-4,
    "note": (
        "A null in an expected array means NaN; a port must produce NaN there too. "
        "Mismatched validity is a hard failure regardless of tolerance."
    ),
}


def _resolve_project_path(path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _jsonable(value):
    """Recursively convert NumPy values to JSON, mapping non-finite floats to None.

    ``json.dumps`` would otherwise emit bare ``NaN``/``Infinity``, which is not
    valid JSON and which strict parsers (Swift's ``JSONDecoder`` among them)
    reject outright.
    """
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value]
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    # Booleans MUST be tested before ints. ``isinstance(True, int)`` is True in
    # Python, so an int-first ordering silently emits ``use_mirror`` as 1/0, and
    # a strict consumer decoding it as a boolean (Swift's ``JSONDecoder``) then
    # rejects the file. np.bool_ is not an int subclass, so only plain bool was
    # ever affected -- which is exactly the type ``use_mirror`` has.
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def load_fixtures(
    keypoints_dir: Path, max_fixtures: int | None, include_short_windows: bool = False
) -> list[dict]:
    """Load the committed keypoint windows, trimmed the way ``run()`` trims them.

    The live trainer only ever scores a window of exactly
    ``num_video_sequence_samples`` frames: ``run()`` gates on
    ``len(stacked_seq) >= required_capture_frames`` (action_recognition.py:2803),
    and in the default reference mode ``required_capture_frames`` *is*
    ``num_video_sequence_samples`` (:2789-2792). So a shorter window is not a
    useful edge case -- it is an input the app cannot produce.

    Runtime-reachable fixtures therefore come first and are the ones that get
    full-array coverage. ``include_short_windows`` additionally exports the
    shorter committed tracks, flagged ``runtime_reachable: false``, to exercise
    the degenerate branches in ``resample_pose_sequence`` (T=1 hits its
    single-sample path) for a port that wants to match those too.
    """
    track_files = sorted(keypoints_dir.glob("track_*.npy"))
    if not track_files:
        raise SystemExit(f"no keypoint tracks found in {keypoints_dir}")

    reachable: list[dict] = []
    short: list[dict] = []
    for fp in track_files:
        try:
            arr = np.load(fp)
        except Exception as exc:
            print(f"warning: skipping {fp.name}: {exc}")
            continue
        if arr.ndim != 3 or arr.shape[1] != 17 or arr.shape[2] < 3:
            print(f"warning: skipping {fp.name} with shape {arr.shape}")
            continue
        window = arr[-NUM_VIDEO_SEQUENCE_SAMPLES:].astype(np.float32)
        entry = {
            "name": fp.stem,
            "window": window,
            "runtime_reachable": bool(window.shape[0] >= NUM_VIDEO_SEQUENCE_SAMPLES),
        }
        (reachable if entry["runtime_reachable"] else short).append(entry)

    if not reachable:
        raise SystemExit(
            f"no runtime-reachable ({NUM_VIDEO_SEQUENCE_SAMPLES}-frame) windows in {keypoints_dir}"
        )

    ordered = list(reachable)
    if include_short_windows:
        ordered += short
    else:
        print(
            f"note: {len(short)} committed tracks are shorter than "
            f"{NUM_VIDEO_SEQUENCE_SAMPLES} frames and cannot occur at runtime; excluded. "
            "Pass --include-short-windows to export them anyway."
        )

    if max_fixtures is not None:
        ordered = ordered[:max_fixtures]
    return ordered


def level_normalize(fixtures: list[dict]) -> dict:
    """``normalize_pose_sequence`` output per fixture."""
    return {
        f["name"]: _jsonable(normalize_pose_sequence(f["window"], conf_thresh=CONF_THRESH))
        for f in fixtures
    }


def level_resample(fixtures: list[dict]) -> dict:
    """``resample_pose_sequence`` output per fixture, at each target length."""
    out: dict[str, dict] = {}
    for f in fixtures:
        norm = normalize_pose_sequence(f["window"], conf_thresh=CONF_THRESH)
        out[f["name"]] = {
            str(target): _jsonable(resample_pose_sequence(norm, target))
            for target in RESAMPLE_TARGET_LENGTHS
        }
    return out


#: Full cost matrices above this many cells are summarized instead of dumped.
#: The longest reference is 196 frames, so an unabridged matrix there would be
#: ~38k floats -- more JSON than the diagnostic value justifies.
FULL_COST_MATRIX_MAX_CELLS = 1024


def _reference_length_spread(references: dict, techniques: list[str]) -> list[tuple[str, str]]:
    """Shortest, median, and longest reference in the library, as (technique, angle).

    The DTW level covers a spread of sequence lengths rather than whichever
    technique sorts first: matrix size is the dimension a DTW port is most likely
    to get wrong, and one length exercises none of it.
    """
    lengths = [
        (int(references[t][a].shape[0]), t, a)
        for t in techniques
        for a in sorted(references.get(t, {}))
    ]
    if not lengths:
        return []
    lengths.sort()
    picks = {lengths[0], lengths[len(lengths) // 2], lengths[-1]}
    return [(t, a) for _, t, a in sorted(picks)]


def level_dtw(fixtures: list[dict], references: dict, techniques: list[str]) -> list[dict]:
    """Cost matrix and DTW distance for each fixture against a spread of references.

    The most diagnostic level for a port: a wrong cost matrix and a wrong DTW
    recurrence produce the same end-to-end symptom but are separated here.

    Note the invariant this reveals -- ``compare_pose_sequence`` resamples the
    user window to the reference's own ``target_len``, so ``na == nb`` always and
    the cost matrix is always square. A port never has to handle the rectangular
    case, whatever the general DTW recurrence allows.

    Matrices small enough to read are dumped whole; larger ones are fingerprinted
    by their row and column sums, which localize a divergence to a frame at
    linear cost instead of quadratic.
    """
    out: list[dict] = []
    targets = _reference_length_spread(references, techniques)
    for f in fixtures:
        norm = normalize_pose_sequence(f["window"], conf_thresh=CONF_THRESH)
        for technique, angle in targets:
            prepared = _prepare_reference_sequence(references[technique][angle], conf_thresh=CONF_THRESH)
            usr_res = resample_pose_sequence(norm, prepared["target_len"])
            ref_res = prepared["ref_res"]
            cost = _pairwise_frame_cost_matrix(usr_res, ref_res)

            case = {
                "fixture": f["name"],
                "technique": technique,
                "angle": angle,
                "user_resampled_len": int(usr_res.shape[0]),
                "reference_resampled_len": int(ref_res.shape[0]),
                "dtw_min_plus": _jsonable(_dtw_min_plus(cost)),
                "dtw_pose_distance": _jsonable(dtw_pose_distance(usr_res, ref_res)),
                "mirrored_dtw_pose_distance": _jsonable(
                    dtw_pose_distance(_mirror_sequence(usr_res), ref_res)
                ),
            }
            if cost.size <= FULL_COST_MATRIX_MAX_CELLS:
                case["cost_matrix"] = _jsonable(cost)
            else:
                # Marginals alone are NOT injective: a compensating perturbation
                # (e.g. +d, -d, -d, +d on the corners of a rectangle) leaves both
                # row and column sums, the total, the min, the max AND the DTW
                # scalar bit-identical. Pin the diagonal (which mean_pose_distance
                # is read off) and an exact-bit checksum too, so a port cannot
                # land in that null space undetected.
                case["cost_matrix_row_sums"] = _jsonable(cost.sum(axis=1))
                case["cost_matrix_col_sums"] = _jsonable(cost.sum(axis=0))
                case["cost_matrix_diagonal"] = _jsonable(np.diagonal(cost))
                case["cost_matrix_stats"] = {
                    "sum": _jsonable(cost.sum()),
                    "min": _jsonable(cost.min()),
                    "max": _jsonable(cost.max()),
                    "abs_sum": _jsonable(np.abs(cost).sum()),
                    "sum_of_squares": _jsonable(float((cost.astype(np.float64) ** 2).sum())),
                }
                case["cost_matrix_sha256"] = hashlib.sha256(
                    np.ascontiguousarray(cost, dtype="<f4").tobytes()
                ).hexdigest()
            out.append(case)
    return out


def level_compare(fixtures: list[dict], references: dict, techniques: list[str]) -> list[dict]:
    """``compare_pose_sequence`` metric bundle for every (fixture, technique, angle).

    Scalars only, so this is the widest parity surface available cheaply.
    """
    out: list[dict] = []
    for f in fixtures:
        for technique in techniques:
            bank = references.get(technique)
            if not bank:
                continue
            for angle in sorted(bank):
                metrics = compare_pose_sequence(f["window"], bank[angle], technique, conf_thresh=CONF_THRESH)
                out.append(
                    {
                        "fixture": f["name"],
                        "technique": technique,
                        "angle": angle,
                        "metrics": _jsonable(metrics),
                    }
                )
    return out


def level_best_match(fixtures: list[dict], references: dict, techniques: list[str]) -> list[dict]:
    """``_best_reference_match`` chosen angle and metrics, plus the topk prescreen.

    ``topk`` is recorded alongside the exhaustive result because ``--score-topk``
    is a supported runtime mode and a port that implements the prescreen must
    reproduce its selection, not just the full scan.
    """
    out: list[dict] = []
    for f in fixtures:
        for technique in techniques:
            bank = references.get(technique)
            if not bank:
                continue
            full = _best_reference_match(f["window"], bank, technique, conf_thresh=CONF_THRESH)
            topk = _best_reference_match(f["window"], bank, technique, conf_thresh=CONF_THRESH, topk=3)
            out.append(
                {
                    "fixture": f["name"],
                    "technique": technique,
                    "angle_category": _technique_angle_category(technique),
                    "reference_count": len(bank),
                    "best_angle": full[0] if full else None,
                    "metrics": _jsonable(full[1]) if full else None,
                    "best_angle_topk3": topk[0] if topk else None,
                    "metrics_topk3": _jsonable(topk[1]) if topk else None,
                }
            )
    return out


def level_feedback(fixtures: list[dict], references: dict, techniques: list[str]) -> list[dict]:
    """``generate_feedback`` strings at the score the best match actually produced."""
    out: list[dict] = []
    for f in fixtures:
        for technique in techniques:
            bank = references.get(technique)
            if not bank:
                continue
            match = _best_reference_match(f["window"], bank, technique, conf_thresh=CONF_THRESH)
            score = float(match[1]["score"]) if match else 0.0
            out.append(
                {
                    "fixture": f["name"],
                    "technique": technique,
                    "score": _jsonable(score),
                    "feedback": generate_feedback(technique, f["window"], score),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--reference-dir", default="reference_poses", help="reference pose root (default: %(default)s)"
    )
    parser.add_argument(
        "--keypoints-dir", default="keypoints", help="committed fixture windows (default: %(default)s)"
    )
    parser.add_argument(
        "--out", default="export/mobile/golden_vectors.json", help="output path (default: %(default)s)"
    )
    parser.add_argument(
        "--max-fixtures", type=int, default=None, help="limit fixtures (default: all)"
    )
    parser.add_argument(
        "--include-short-windows",
        action="store_true",
        help="also export committed tracks shorter than the 8-frame runtime window (cannot occur live)",
    )
    parser.add_argument(
        "--array-fixtures",
        type=int,
        default=3,
        help="fixtures given full-array coverage at the normalize/resample/dtw levels (default: %(default)s)",
    )
    parser.add_argument(
        "--technique", action="append", default=None, help="limit to specific technique(s); repeatable"
    )
    args = parser.parse_args()

    reference_dir = _resolve_project_path(args.reference_dir)
    keypoints_dir = _resolve_project_path(args.keypoints_dir)
    out_path = _resolve_project_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    references = load_reference_pose_library(str(reference_dir))
    if not references:
        raise SystemExit(f"no references loaded from {reference_dir}")

    techniques = sorted(args.technique or references.keys())
    missing = [t for t in techniques if t not in references]
    for t in missing:
        print(f"warning: technique '{t}' not in reference library; skipped")
    techniques = [t for t in techniques if t in references]

    fixtures = load_fixtures(keypoints_dir, args.max_fixtures, args.include_short_windows)
    array_fixtures = fixtures[: max(0, args.array_fixtures)]

    reachable_count = sum(1 for f in fixtures if f["runtime_reachable"])
    print(
        f"fixtures: {len(fixtures)} ({reachable_count} runtime-reachable, "
        f"{len(array_fixtures)} with full-array coverage)"
    )
    print(f"techniques: {len(techniques)}, references: {sum(len(references[t]) for t in techniques)}")

    golden = {
        "format_version": GOLDEN_FORMAT_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Expected output of the Python scoring core, layered by function, for "
            "verifying a reimplementation. See scripts/export_golden_vectors.py."
        ),
        "null_means_nan": True,
        "conf_thresh": CONF_THRESH,
        "num_video_sequence_samples": NUM_VIDEO_SEQUENCE_SAMPLES,
        "resample_target_lengths": list(RESAMPLE_TARGET_LENGTHS),
        "tolerances": COMPARISON_TOLERANCES,
        "techniques": techniques,
        "inputs": {
            f["name"]: {
                "frames": int(f["window"].shape[0]),
                "runtime_reachable": f["runtime_reachable"],
                "window": _jsonable(f["window"]),
            }
            for f in fixtures
        },
        "levels": {
            "1_normalize": level_normalize(array_fixtures),
            "2_resample": level_resample(array_fixtures),
            "3_dtw": level_dtw(array_fixtures, references, techniques),
            "4_compare": level_compare(fixtures, references, techniques),
            "5_best_match": level_best_match(fixtures, references, techniques),
            "6_feedback": level_feedback(fixtures, references, techniques),
        },
    }

    out_path.write_text(json.dumps(golden, indent=2, allow_nan=False), encoding="utf-8")

    levels = golden["levels"]
    print(f"\nwrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  1_normalize:  {len(levels['1_normalize'])} sequences")
    print(f"  2_resample:   {len(levels['2_resample'])} sequences x {len(RESAMPLE_TARGET_LENGTHS)} lengths")
    print(f"  3_dtw:        {len(levels['3_dtw'])} cost matrices")
    print(f"  4_compare:    {len(levels['4_compare'])} metric bundles")
    print(f"  5_best_match: {len(levels['5_best_match'])} best-match results")
    print(f"  6_feedback:   {len(levels['6_feedback'])} feedback results")


if __name__ == "__main__":
    main()
