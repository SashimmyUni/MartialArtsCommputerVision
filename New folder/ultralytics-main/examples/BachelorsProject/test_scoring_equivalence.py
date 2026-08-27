"""Verify the optimized scoring core in action_recognition.py against a pinned
copy of the original, pre-optimization implementation.

This is the correctness gate for the runtime-speedup work: Phase 2 (caching
normalized/resampled references) and Phase 3 (vectorizing DTW/angle-error) are
supposed to be bit-identical to float rounding, never a behavior change. This
script proves that by running both implementations, side by side, over every
reference under ``reference_poses/`` against a representative set of the
committed ``keypoints/track_*.npy`` windows, and asserting every returned
metric matches within a tight tolerance.

No video required. Run from this directory (or anywhere — paths are resolved
relative to this file):

    python test_scoring_equivalence.py

Exits 0 and prints a summary on success; raises AssertionError with the exact
mismatching value on the first failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import action_recognition as ar  # noqa: E402  (the optimized module under test)

TOL = 1e-4

# ---------------------------------------------------------------------------
# Pinned reference implementation: line-for-line the scoring core as it
# existed before the optimization pass (module-level cache of prepared
# references, vectorized DTW/angle error). Only used here, as the ground
# truth to check the optimized functions against — never used by the app.
# ---------------------------------------------------------------------------


def _legacy_safe_joint(frame: np.ndarray, idx: int) -> np.ndarray | None:
    if idx >= frame.shape[0]:
        return None
    p = frame[idx]
    if p.shape[0] < 2:
        return None
    if not np.isfinite(p[:2]).all():
        return None
    return p[:2]


def _legacy_joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6:
        return float("nan")
    cosang = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _legacy_frame_pose_distance(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if not valid.any():
        return 1.0
    return float(np.mean(np.linalg.norm(a[valid] - b[valid], axis=1)))


def _legacy_dtw_pose_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    na, nb = seq_a.shape[0], seq_b.shape[0]
    dp = np.full((na + 1, nb + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = _legacy_frame_pose_distance(seq_a[i - 1], seq_b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[na, nb] / max(na, nb))


def _legacy_mean_angle_sequence(seq: np.ndarray, a: int, b: int, c: int) -> float:
    vals = []
    for frame in seq:
        pa = _legacy_safe_joint(frame, a)
        pb = _legacy_safe_joint(frame, b)
        pc = _legacy_safe_joint(frame, c)
        if pa is None or pb is None or pc is None:
            continue
        ang = _legacy_joint_angle_deg(pa, pb, pc)
        if np.isfinite(ang):
            vals.append(ang)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _legacy_technique_angle_error(user_seq: np.ndarray, ref_seq: np.ndarray, technique: str) -> float:
    t = technique.lower().replace("-", " ").strip()
    if "jab" in t or "cross" in t or "hook" in t:
        angle_defs = [(5, 7, 9), (6, 8, 10), (7, 5, 11), (8, 6, 12)]
    elif "kick" in t:
        angle_defs = [(11, 13, 15), (12, 14, 16), (5, 11, 13), (6, 12, 14)]
    else:
        angle_defs = [(5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16)]

    errs = []
    for a, b, c in angle_defs:
        u = _legacy_mean_angle_sequence(user_seq, a, b, c)
        r = _legacy_mean_angle_sequence(ref_seq, a, b, c)
        if np.isfinite(u) and np.isfinite(r):
            errs.append(abs(u - r))
    if not errs:
        return 90.0
    return float(np.mean(errs))


def _legacy_cosine_pose_similarity(user_seq: np.ndarray, ref_seq: np.ndarray) -> float:
    u = np.nan_to_num(user_seq, nan=0.0).reshape(-1)
    r = np.nan_to_num(ref_seq, nan=0.0).reshape(-1)
    nu = np.linalg.norm(u)
    nr = np.linalg.norm(r)
    if nu < 1e-6 or nr < 1e-6:
        return 0.0
    return float(np.dot(u, r) / (nu * nr))


def legacy_compare_pose_sequence(
    user_sequence: np.ndarray, reference_sequence: np.ndarray, technique: str, conf_thresh: float = 0.2
) -> dict[str, float | bool]:
    """Exact copy of ``compare_pose_sequence`` as it existed before optimization."""
    ref_norm = ar.normalize_pose_sequence(reference_sequence, conf_thresh=conf_thresh)
    user_norm = ar.normalize_pose_sequence(user_sequence, conf_thresh=conf_thresh)

    target_len = max(4, ref_norm.shape[0])
    ref_res = ar.resample_pose_sequence(ref_norm, target_len)
    usr_res = ar.resample_pose_sequence(user_norm, target_len)
    usr_mirror_res = ar._mirror_sequence(usr_res)

    cos_plain = _legacy_cosine_pose_similarity(usr_res, ref_res)
    cos_mirror = _legacy_cosine_pose_similarity(usr_mirror_res, ref_res)
    use_mirror = cos_mirror > cos_plain
    usr_best = usr_mirror_res if use_mirror else usr_res
    cos_sim = max(cos_plain, cos_mirror)

    dtw_dist = _legacy_dtw_pose_distance(usr_best, ref_res)
    angle_err = _legacy_technique_angle_error(usr_best, ref_res, technique)
    mean_dist = float(np.mean([_legacy_frame_pose_distance(a, b) for a, b in zip(usr_best, ref_res)]))

    cosine_score = (cos_sim + 1.0) * 50.0
    dtw_score = max(0.0, 100.0 * (1.0 - (dtw_dist / 0.8)))
    angle_score = max(0.0, 100.0 * (1.0 - (angle_err / 90.0)))
    pose_dist_score = max(0.0, 100.0 * (1.0 - (mean_dist / 0.8)))
    final_score = 0.35 * cosine_score + 0.25 * dtw_score + 0.25 * angle_score + 0.15 * pose_dist_score

    return {
        "use_mirror": use_mirror,
        "cosine_similarity": float(cos_sim),
        "dtw_distance": float(dtw_dist),
        "angle_error": float(angle_err),
        "mean_pose_distance": float(mean_dist),
        "score": float(np.clip(final_score, 0.0, 100.0)),
    }


# ---------------------------------------------------------------------------
# Equivalence checks
# ---------------------------------------------------------------------------


def _assert_close(name: str, legacy_val: object, new_val: object, tol: float = TOL) -> None:
    if isinstance(legacy_val, (bool, np.bool_)) or isinstance(new_val, (bool, np.bool_)):
        assert bool(legacy_val) == bool(new_val), f"{name}: mismatch {legacy_val!r} != {new_val!r}"
        return
    diff = abs(float(legacy_val) - float(new_val))
    assert diff <= tol, f"{name}: {legacy_val!r} vs {new_val!r} (diff {diff:.6g} > tol {tol:g})"


def _compare_metrics(legacy: dict, new: dict, context: str) -> None:
    for key in ("score", "cosine_similarity", "dtw_distance", "angle_error", "mean_pose_distance", "use_mirror"):
        _assert_close(f"{context}.{key}", legacy[key], new[key])


def _select_representative_user_windows(keypoints_dir: Path) -> list[tuple[str, np.ndarray]]:
    """Pick one keypoint track per distinct frame-count (T) seen in
    keypoints/track_*.npy. All comparisons resample the user window to the
    reference's own length before scoring, so the original user T only
    matters insofar as it exercises different code paths in
    ``resample_pose_sequence`` (single-frame fill vs. real interpolation) —
    testing every track file adds runtime without adding coverage.
    """
    track_files = sorted(keypoints_dir.glob("track_*.npy"))
    assert track_files, f"no keypoint tracks found in {keypoints_dir}"

    by_t: dict[int, tuple[str, np.ndarray]] = {}
    for fp in track_files:
        arr = np.load(fp)
        if arr.ndim == 3 and arr.shape[0] >= 1 and arr.shape[1] >= 10:
            by_t.setdefault(int(arr.shape[0]), (fp.name, arr.astype(np.float32)))

    assert by_t, f"no usable (T>=1, K>=10) keypoint windows found in {keypoints_dir}"
    return [by_t[t] for t in sorted(by_t)]


def main() -> None:
    reference_dir = SCRIPT_DIR / "reference_poses"
    keypoints_dir = SCRIPT_DIR / "keypoints"

    references = ar.load_reference_pose_library(str(reference_dir))
    assert references, f"no references loaded from {reference_dir}"
    total_refs = sum(len(bank) for bank in references.values())

    user_windows = _select_representative_user_windows(keypoints_dir)
    print(
        f"testing {total_refs} reference sequences across {len(references)} techniques "
        f"against {len(user_windows)} representative user windows "
        f"(T = {[w.shape[0] for _, w in user_windows]})"
    )

    pair_checks = 0
    best_match_checks = 0

    for technique, bank in references.items():
        for user_name, user_seq in user_windows:
            legacy_metrics_by_angle: dict[str, dict] = {}
            for angle, ref_seq in bank.items():
                legacy = legacy_compare_pose_sequence(user_seq, ref_seq, technique)
                new = ar.compare_pose_sequence(user_seq, ref_seq, technique)
                _compare_metrics(legacy, new, f"{technique}/{angle} vs {user_name}")
                legacy_metrics_by_angle[angle] = legacy
                pair_checks += 1

            # Derive the legacy "best" selection from the metrics already computed
            # above (same strict->, first-seen-wins tie-break as the original
            # _best_reference_match) instead of recomputing every pair again.
            legacy_best_angle = ""
            legacy_best_metrics = None
            legacy_best_score = -1.0
            for angle in bank:  # preserve original dict order for tie-breaking
                m = legacy_metrics_by_angle[angle]
                if float(m["score"]) > legacy_best_score:
                    legacy_best_score = float(m["score"])
                    legacy_best_angle = angle
                    legacy_best_metrics = m

            new_best = ar._best_reference_match(user_seq, bank, technique)
            assert new_best is not None, f"{technique} vs {user_name}: new _best_reference_match returned None"
            new_angle, new_metrics = new_best
            assert legacy_best_angle == new_angle, (
                f"{technique} vs {user_name}: best angle mismatch {legacy_best_angle!r} != {new_angle!r}"
            )
            _compare_metrics(legacy_best_metrics, new_metrics, f"{technique} best-match vs {user_name}")
            best_match_checks += 1

    print(f"OK: {pair_checks} pairwise compare_pose_sequence checks, {best_match_checks} best_reference_match checks")
    print(f"all metrics matched within tolerance {TOL:g}")


if __name__ == "__main__":
    main()
