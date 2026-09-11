"""Verify the exported port artifacts are self-contained and reproduce the core.

Stands in for the test harness a Swift or TypeScript port will write. It loads
**only** ``export/mobile/`` -- never ``reference_poses/*.npy`` or
``keypoints/*.npy`` -- rebuilds the references from the binary blob and the input
windows from the embedded lists, re-runs the Python scoring core on them, and
checks the results against the recorded expectations.

If this passes, three things are true:

1. The bundle and golden vectors are genuinely self-contained; a port needs
   nothing else from this repo.
2. The recorded expectations actually match what the scoring core produces, so a
   port that reproduces them is correct rather than bug-compatible with a stale
   export.
3. The golden JSON parses under strict rules, with no bare ``NaN``/``Infinity``
   that would break a non-Python consumer.

Run it after regenerating either export, and treat a failure as a broken export
rather than a broken port:

    python scripts/export_mobile_bundle.py
    python scripts/export_golden_vectors.py
    python scripts/verify_export.py

Exits non-zero on any failure, so it works as a CI step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# action_recognition.py lives at the repo root, one level above this script.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from action_recognition import (  # noqa: E402
    COCO17_EDGES,
    _ANGLE_DEF_CATEGORIES,
    _best_reference_match,
    _dtw_min_plus,
    _mirror_sequence,
    _pairwise_frame_cost_matrix,
    _prepare_reference_sequence,
    _technique_angle_category,
    compare_pose_sequence,
    dtw_pose_distance,
    generate_feedback,
    normalize_pose_sequence,
    resample_pose_sequence,
)

#: Row/column sums accumulate over a whole axis, so they need a looser bound than
#: the individual cells they are derived from. A 196-cell row of values under ~1.0
#: accumulates at most ~196 * 1e-5 of float32 error, so 1e-3 is the right order.
COST_MARGINAL_ATOL = 1e-3

#: Joint indices the scoring core hardcodes. Checked against the manifest's
#: keypoint_names so a reordered or mislabelled export cannot pass: the scorer
#: reads shoulders at 5/6 (normalize_pose_frame scale), hips at 11/12 (its
#: centre), wrists at 9/10 (_wrist_motion_energy), and the angle triplets use
#: elbows 7/8, knees 13/14, ankles 15/16.
EXPECTED_JOINTS = {
    0: "nose",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

#: Score-weight split from the formula at action_recognition.py:809-817.
EXPECTED_WEIGHTS = {"cosine": 0.35, "dtw": 0.25, "angle": 0.25, "pose_dist": 0.15}

#: Every other constant the manifest publishes as the porting spec, with the value
#: the live code actually uses. A wrong value here silently corrupts a port, so
#: these are compared rather than assumed.
EXPECTED_CONSTANTS = {
    "conf_thresh": 0.2,
    "num_video_sequence_samples": 8,
    "min_resample_len": 4,
    "dtw_score_divisor": 0.8,
    "angle_score_divisor": 90.0,
    "pose_dist_score_divisor": 0.8,
    "mirror_swaps_left_right_indices": False,
    "mean_pose_distance_is_cost_matrix_diagonal": True,
    "dtw_distance_normalized_by_matrix_dimension": True,
    "cost_matrix_is_always_square": True,
    "array_dtype": "float32",
    "score_arithmetic_dtype": "float64",
}


class Report:
    """Collects failures so one run surfaces every problem, not just the first."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        print(f"  FAIL {msg}")

    def ok(self, msg: str) -> None:
        print(f"  {msg}")


def _resolve_project_path(path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def to_array(nested) -> np.ndarray:
    """Nested lists with null-for-NaN back into a ``(T,K,C)`` float32 array."""
    return np.array(
        [[[np.nan if v is None else v for v in joint] for joint in frame] for frame in nested],
        dtype=np.float32,
    )


def check_strict_json(golden_text: str, report: Report) -> None:
    """Reject bare NaN/Infinity, which strict parsers refuse.

    Python's ``json`` accepts them; Swift's ``JSONDecoder`` does not.
    ``parse_constant`` fires on exactly those tokens in a value position, so
    raising from it is the real check -- scanning the text for the substring would
    just match the word "NaN" in the file's own documentation prose.
    """

    def reject(name: str):
        raise ValueError(f"non-standard JSON constant in value position: {name}")

    try:
        json.loads(golden_text, parse_constant=reject)
        report.ok("strict JSON: no bare NaN/Infinity in any value position")
    except ValueError as exc:
        report.fail(str(exc))


def close_enough(actual, want, atol: float) -> bool:
    """Tolerance comparison that does NOT silently pass on NaN.

    ``abs(float('nan') - x) > atol`` evaluates False, so the naive form treats a
    core that returns NaN where the golden file records a finite number as a
    pass. Validity has to be compared before magnitude.
    """
    a, w = float(actual), float(want)
    if math.isnan(a) != math.isnan(w):
        return False
    if math.isnan(a) and math.isnan(w):
        return True
    if math.isinf(a) or math.isinf(w):
        return a == w
    return abs(a - w) <= atol


def check_blob_integrity(manifest: dict, blob: bytes, report: Report) -> None:
    """Validate the blob as a container before trusting any offset in it.

    Each of these has been wrong in a real export at some point, and none of them
    is caught by comparing scores: a gapped or overlapping offset table, a
    byte_length inconsistent with the declared frame count, a blob the manifest
    only partly accounts for, or a payload that does not match its own checksum.
    """
    if "total_bytes" in manifest and len(blob) != manifest["total_bytes"]:
        report.fail(f"blob is {len(blob)} bytes, manifest total_bytes says {manifest['total_bytes']}")

    declared = manifest.get("data_sha256")
    if not declared:
        report.fail("manifest carries no data_sha256")
    elif hashlib.sha256(blob).hexdigest() != declared:
        report.fail("blob sha256 does not match manifest data_sha256")

    spans = []
    for technique, meta in manifest["techniques"].items():
        for angle, entry in meta["references"].items():
            expected = entry["frames"] * 17 * 3 * 4  # float32
            if entry["byte_length"] != expected:
                report.fail(
                    f"{technique}/{angle} byte_length {entry['byte_length']} != "
                    f"frames*17*3*4 = {expected}"
                )
            if entry["offset"] % 4:
                report.fail(f"{technique}/{angle} offset {entry['offset']} is not float32-aligned")
            if entry["offset"] + entry["byte_length"] > len(blob):
                report.fail(f"{technique}/{angle} extends past the end of the blob")
            spans.append((entry["offset"], entry["offset"] + entry["byte_length"], f"{technique}/{angle}"))

    spans.sort()
    covered = 0
    for i, (start, end, name) in enumerate(spans):
        covered += end - start
        if i and start < spans[i - 1][1]:
            report.fail(f"{name} overlaps {spans[i - 1][2]} in the blob")
        elif i and start > spans[i - 1][1]:
            report.fail(f"{start - spans[i - 1][1]} unaccounted bytes between {spans[i - 1][2]} and {name}")
    if covered != len(blob):
        report.fail(f"references account for {covered} of {len(blob)} blob bytes")
    else:
        report.ok(f"blob integrity: {len(spans)} spans, contiguous, checksum matches, fully consumed")


def load_references_from_blob(manifest: dict, blob: bytes, report: Report) -> dict:
    """Rebuild the reference library from the manifest offsets and the blob."""
    references: dict[str, dict[str, np.ndarray]] = {}
    for technique, meta in manifest["techniques"].items():
        for angle, entry in meta["references"].items():
            if entry["offset"] + entry["byte_length"] > len(blob):
                continue  # already reported by check_blob_integrity
            arr = np.frombuffer(
                blob, dtype="<f4", count=entry["frames"] * 17 * 3, offset=entry["offset"]
            ).reshape(entry["frames"], 17, 3)
            references.setdefault(technique, {})[angle] = np.ascontiguousarray(arr)
    return references


def cmp_array(label: str, got: np.ndarray, want_nested, atol: float, report: Report) -> None:
    """Compare against an expected array in which null means NaN.

    Validity must match exactly: a number where NaN is expected, or the reverse,
    is a hard failure regardless of tolerance, because NaN is precisely how the
    pipeline excludes occluded joints from a comparison.
    """
    want = to_array(want_nested)
    if got.shape != want.shape:
        report.fail(f"{label} shape {got.shape} vs {want.shape}")
        return
    got_nan, want_nan = ~np.isfinite(got), ~np.isfinite(want)
    if not np.array_equal(got_nan, want_nan):
        report.fail(f"{label} NaN mask differs in {int((got_nan != want_nan).sum())} cells")
        return
    valid = ~got_nan
    if valid.any():
        diff = float(np.max(np.abs(got[valid] - want[valid])))
        if diff > atol:
            report.fail(f"{label} max abs diff {diff:.3e} > {atol:.0e}")


def check_manifest(manifest: dict, report: Report) -> None:
    """Validate the manifest against the LIVE code, not against itself.

    The manifest is the actual porting spec: a port reads its constants, joint
    order and angle triplets and never sees this repo. Self-consistency checks are
    therefore worthless here -- an entirely wrong manifest is self-consistent. Every
    check below compares a manifest value against the thing in
    ``action_recognition.py`` it is supposed to mirror.
    """
    required = {
        "keypoint_names",
        "skeleton_edges",
        "angle_definitions",
        "scoring_constants",
        "coordinate_space",
        "techniques",
    }
    missing = required - set(manifest)
    if missing:
        report.fail(f"manifest missing keys: {sorted(missing)}")
        return

    # --- joint order, checked against the indices the scoring core actually uses
    names = manifest["keypoint_names"]
    if len(names) != 17:
        report.fail(f"keypoint_names has {len(names)} entries, expected 17")
    else:
        for idx, expected in EXPECTED_JOINTS.items():
            if names[idx] != expected:
                report.fail(f"keypoint_names[{idx}] is '{names[idx]}', must be '{expected}'")

    # --- skeleton edges and angle triplets, checked against the live constants
    if [list(e) for e in COCO17_EDGES] != [list(e) for e in manifest["skeleton_edges"]]:
        report.fail("skeleton_edges does not match COCO17_EDGES in action_recognition.py")

    live_angles = {k: [list(t) for t in v] for k, v in _ANGLE_DEF_CATEGORIES.items()}
    if live_angles != {k: [list(t) for t in v] for k, v in manifest["angle_definitions"].items()}:
        report.fail("angle_definitions does not match _ANGLE_DEF_CATEGORIES in action_recognition.py")

    # --- scoring constants, each checked against the source of truth
    constants = manifest["scoring_constants"]
    for key, expected in EXPECTED_CONSTANTS.items():
        if key not in constants:
            report.fail(f"scoring_constants missing '{key}'")
        elif constants[key] != expected:
            report.fail(f"scoring_constants['{key}'] is {constants[key]!r}, expected {expected!r}")

    weights = constants.get("score_weights", {})
    if weights != EXPECTED_WEIGHTS:
        report.fail(f"score_weights {weights} does not match the formula at action_recognition.py:809-817")
    elif abs(sum(weights.values()) - 1.0) > 1e-9:
        report.fail(f"score_weights sum to {sum(weights.values())}, expected 1.0")

    # --- mirror semantics, established by running the function rather than asserting
    probe = np.zeros((1, 17, 2), dtype=np.float32)
    probe[0, 5] = (1.0, 2.0)   # left shoulder
    probe[0, 6] = (-1.0, 3.0)  # right shoulder
    mirrored = _mirror_sequence(probe)
    swaps_indices = bool(np.allclose(mirrored[0, 5], (1.0, 3.0)))
    negates_x_only = bool(
        np.allclose(mirrored[0, 5], (-1.0, 2.0)) and np.allclose(mirrored[0, 6], (1.0, 3.0))
    )
    if not negates_x_only or swaps_indices:
        report.fail("_mirror_sequence behaviour changed; the manifest's mirror flag is now wrong")
    elif constants.get("mirror_swaps_left_right_indices") is not False:
        report.fail(
            "scoring_constants.mirror_swaps_left_right_indices must be false: "
            "_mirror_sequence was measured to negate x only"
        )

    # --- per-technique angle category, checked against the live classifier
    for technique, meta in manifest["techniques"].items():
        if meta["angle_category"] not in manifest["angle_definitions"]:
            report.fail(f"{technique} references unknown angle_category '{meta['angle_category']}'")
        live = _technique_angle_category(technique)
        if meta["angle_category"] != live:
            report.fail(
                f"{technique} angle_category is '{meta['angle_category']}' but "
                f"_technique_angle_category says '{live}' - the export is stale"
            )

    report.ok(
        f"manifest: {len(manifest['techniques'])} techniques, "
        f"{sum(len(m['references']) for m in manifest['techniques'].values())} references, "
        "constants/edges/triplets/categories all cross-checked against live code"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--export-dir", default="export/mobile", help="directory holding the exports (default: %(default)s)"
    )
    args = parser.parse_args()

    export_dir = _resolve_project_path(args.export_dir)
    golden_path = export_dir / "golden_vectors.json"
    manifest_path = export_dir / "reference_bundle.json"
    blob_path = export_dir / "reference_bundle.bin"

    for path in (golden_path, manifest_path, blob_path):
        if not path.exists():
            raise SystemExit(
                f"missing {path}\nRun scripts/export_mobile_bundle.py and "
                "scripts/export_golden_vectors.py first."
            )

    report = Report()

    golden_text = golden_path.read_text(encoding="utf-8")
    check_strict_json(golden_text, report)
    golden = json.loads(golden_text)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blob = blob_path.read_bytes()

    check_manifest(manifest, report)

    check_blob_integrity(manifest, blob, report)
    references = load_references_from_blob(manifest, blob, report)
    report.ok(f"rebuilt {sum(len(b) for b in references.values())} references from the blob alone")

    inputs = {name: to_array(spec["window"]) for name, spec in golden["inputs"].items()}
    report.ok(f"rebuilt {len(inputs)} input windows from the embedded lists")

    conf = golden["conf_thresh"]
    tol = golden["tolerances"]
    levels = golden["levels"]

    # ------------------------------------------------------------- normalize
    for name, want in levels["1_normalize"].items():
        cmp_array(
            f"1_normalize {name}",
            normalize_pose_sequence(inputs[name], conf_thresh=conf),
            want,
            tol["normalized_coordinates_atol"],
            report,
        )
    report.ok(f"1_normalize:  {len(levels['1_normalize'])} sequences reproduced")

    # -------------------------------------------------------------- resample
    count = 0
    for name, by_length in levels["2_resample"].items():
        norm = normalize_pose_sequence(inputs[name], conf_thresh=conf)
        for target, want in by_length.items():
            cmp_array(
                f"2_resample {name}@{target}",
                resample_pose_sequence(norm, int(target)),
                want,
                tol["resampled_coordinates_atol"],
                report,
            )
            count += 1
    report.ok(f"2_resample:   {count} sequences reproduced")

    # ------------------------------------------------------------------- dtw
    for case in levels["3_dtw"]:
        label = f"3_dtw {case['fixture']}/{case['technique']}/{case['angle']}"
        norm = normalize_pose_sequence(inputs[case["fixture"]], conf_thresh=conf)
        prepared = _prepare_reference_sequence(
            references[case["technique"]][case["angle"]], conf_thresh=conf
        )
        usr_res = resample_pose_sequence(norm, prepared["target_len"])
        ref_res = prepared["ref_res"]
        cost = _pairwise_frame_cost_matrix(usr_res, ref_res)

        # The pipeline resamples the user window to the reference's own length,
        # so this should hold for every comparison the app ever performs.
        if usr_res.shape[0] != ref_res.shape[0]:
            report.fail(f"{label} cost matrix not square ({usr_res.shape[0]}x{ref_res.shape[0]})")

        for key, actual in (
            ("dtw_min_plus", _dtw_min_plus(cost)),
            ("dtw_pose_distance", dtw_pose_distance(usr_res, ref_res)),
            ("mirrored_dtw_pose_distance", dtw_pose_distance(_mirror_sequence(usr_res), ref_res)),
        ):
            if not close_enough(actual, case[key], tol["dtw_distance_atol"]):
                report.fail(f"{label} {key}: {actual} vs {case[key]}")

        if "cost_matrix" in case:
            want = np.array(case["cost_matrix"], dtype=np.float32)
            if cost.shape != want.shape:
                report.fail(f"{label} cost_matrix shape {cost.shape} vs {want.shape}")
            elif float(np.max(np.abs(cost - want))) > tol["cost_matrix_atol"]:
                report.fail(f"{label} cost_matrix differs")
        else:
            for key, actual in (
                ("cost_matrix_row_sums", cost.sum(axis=1)),
                ("cost_matrix_col_sums", cost.sum(axis=0)),
            ):
                want = np.array(case[key], dtype=np.float32)
                if actual.shape != want.shape:
                    report.fail(f"{label} {key} shape {actual.shape} vs {want.shape}")
                elif float(np.max(np.abs(actual - want))) > COST_MARGINAL_ATOL:
                    report.fail(f"{label} {key} differs")
    report.ok(f"3_dtw:        {len(levels['3_dtw'])} cost matrices reproduced")

    # --------------------------------------------------------------- compare
    for case in levels["4_compare"]:
        label = f"4_compare {case['fixture']}/{case['technique']}/{case['angle']}"
        got = compare_pose_sequence(
            inputs[case["fixture"]],
            references[case["technique"]][case["angle"]],
            case["technique"],
            conf_thresh=conf,
        )
        for key, want in case["metrics"].items():
            actual = got[key]
            if isinstance(want, bool) or isinstance(actual, (bool, np.bool_)):
                if bool(want) != bool(actual):
                    report.fail(f"{label} {key}: {actual} vs {want}")
            elif want is None:
                if np.isfinite(float(actual)):
                    report.fail(f"{label} {key}: expected NaN, got {actual}")
            elif not close_enough(actual, want, tol["score_atol"]):
                report.fail(f"{label} {key}: {actual} vs {want}")
    report.ok(f"4_compare:    {len(levels['4_compare'])} metric bundles reproduced")

    # ------------------------------------------------------------ best match
    for case in levels["5_best_match"]:
        label = f"5_best_match {case['fixture']}/{case['technique']}"
        got = _best_reference_match(
            inputs[case["fixture"]], references[case["technique"]], case["technique"], conf_thresh=conf
        )
        if got is None:
            if case["best_angle"] is not None:
                report.fail(f"{label}: expected a match, got none")
            continue
        if got[0] != case["best_angle"]:
            report.fail(f"{label} angle: {got[0]} vs {case['best_angle']}")
        if not close_enough(got[1]["score"], case["metrics"]["score"], tol["score_atol"]):
            report.fail(f"{label} score: {got[1]['score']} vs {case['metrics']['score']}")
    report.ok(f"5_best_match: {len(levels['5_best_match'])} best-match results reproduced")

    # -------------------------------------------------------------- feedback
    for case in levels["6_feedback"]:
        got = generate_feedback(case["technique"], inputs[case["fixture"]], float(case["score"]))
        if got != case["feedback"]:
            report.fail(
                f"6_feedback {case['fixture']}/{case['technique']}: {got} vs {case['feedback']}"
            )
    report.ok(f"6_feedback:   {len(levels['6_feedback'])} feedback results reproduced")

    print()
    if report.failures:
        print(f"FAIL: {len(report.failures)} failures")
        raise SystemExit(1)
    print("PASS: exports are self-contained and reproduce the scoring core")


if __name__ == "__main__":
    main()
