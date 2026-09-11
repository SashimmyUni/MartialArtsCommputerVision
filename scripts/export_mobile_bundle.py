"""Export the reference pose library as a self-describing bundle for a port.

``action_recognition.py`` loads references straight off disk as 110 separate
``reference_poses/<technique>/<angle>.npy`` files and derives everything else it
needs (which joint-angle triplets apply to a technique, the per-label score
threshold) from the Python catalogue at runtime. A Swift or TypeScript port has
neither NumPy's ``.npy`` reader nor ``technique_catalog``, so this script
flattens all of it into two files that any language can consume:

- ``reference_bundle.json`` -- manifest: per-technique/angle offsets into the
  blob, the joint-angle definitions, the COCO-17 joint names and skeleton edges,
  and the scoring constants, so the port needs no knowledge of this repo.
- ``reference_bundle.bin`` -- every reference sequence concatenated as
  little-endian float32 ``(T,17,3)`` = ``(x, y, confidence)`` in source pixels.

Sequences are exported **raw**, exactly as ``load_reference_pose_library`` yields
them. Normalization and resampling stay on the consumer side so both
implementations share one input and can be compared frame for frame -- see
``export_golden_vectors.py`` for the harness that does the comparing.

Run from anywhere; paths resolve against the repo root:

    python scripts/export_mobile_bundle.py
    python scripts/export_mobile_bundle.py --format json   # inline, for eyeballing
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    COCO17_EDGES,
    _ANGLE_DEF_CATEGORIES,
    _load_label_thresholds,
    _normalize_key,
    _technique_angle_category,
    load_reference_pose_library,
)

#: Bump when the manifest layout changes in a way a consumer must notice.
BUNDLE_FORMAT_VERSION = 1

#: COCO-17 joint order, matching the index order every reference `.npy` uses and
#: the indices hardcoded throughout the scoring core (5/6 shoulders, 11/12 hips,
#: 9/10 wrists, 13/14 knees, 15/16 ankles). A port maps its own detector's joints
#: onto this order.
COCO17_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

#: Constants baked into the scoring core that a port must match exactly to
#: reproduce scores. Mirrors ``_compare_resampled`` (action_recognition.py:764-817)
#: and ``normalize_pose_frame`` (action_recognition.py:1138).
#:
#: Several entries record *semantics* rather than numbers, because matching every
#: number and still computing a different score is the failure mode this bundle
#: exists to prevent:
#:
#: - ``mirror_swaps_left_right_indices`` is False deliberately: ``_mirror_sequence``
#:   (action_recognition.py:1349) negates x only and does not relabel left/right
#:   joints. Every calibrated threshold assumes that, so reproduce it rather than
#:   "correct" it. Swapping the indices instead costs ~49 score points.
#: - ``mean_pose_distance`` is the **diagonal** of the cost matrix
#:   (action_recognition.py:796, ``np.diagonal(cost).mean()``), NOT the mean cost
#:   along the DTW warping path. This is the single least guessable part of the
#:   formula.
#: - ``dtw_distance`` is divided by the matrix dimension (:795), so it depends on
#:   the cost matrix being square -- which it always is, because the user window is
#:   resampled up to the reference length.
#: - The arrays are float32, but the final score arithmetic runs in Python floats,
#:   i.e. **float64**. A port using Float throughout will not reproduce the
#:   recorded scores; use Float for the arrays and Double for the score algebra.
SCORING_CONSTANTS = {
    "conf_thresh": 0.2,
    "num_video_sequence_samples": 8,
    "min_resample_len": 4,
    "dtw_score_divisor": 0.8,
    "angle_score_divisor": 90.0,
    "pose_dist_score_divisor": 0.8,
    "score_weights": {"cosine": 0.35, "dtw": 0.25, "angle": 0.25, "pose_dist": 0.15},
    "cosine_score_formula": "(cosine_similarity + 1.0) * 50.0",
    "score_clamped_to": [0.0, 100.0],
    "mirror_swaps_left_right_indices": False,
    "mean_pose_distance_is_cost_matrix_diagonal": True,
    "dtw_distance_normalized_by_matrix_dimension": True,
    "cost_matrix_is_always_square": True,
    "resample_fills_nan_before_comparison": True,
    "array_dtype": "float32",
    "score_arithmetic_dtype": "float64",
}


def _resolve_project_path(path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _shared_manifest_fields() -> dict:
    """Manifest keys that do not depend on the reference data itself."""
    return {
        "format_version": BUNDLE_FORMAT_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "contents_note": (
            "Reference pose sequences only. Contains no Ultralytics code or weights."
        ),
        "keypoint_format": "COCO17",
        "keypoint_names": COCO17_KEYPOINT_NAMES,
        "skeleton_edges": [list(edge) for edge in COCO17_EDGES],
        "coordinate_space": {
            "units": "source video pixels",
            "origin": "top-left",
            "y_axis": "down",
            "channels": ["x", "y", "confidence"],
            "note": (
                "Detectors returning normalized [0,1] coordinates with a bottom-left "
                "origin (Apple Vision, for one) must flip y and multiply back by "
                "(frame_width, frame_height) to restore aspect ratio before scoring."
            ),
        },
        "angle_definitions": {
            category: [list(triplet) for triplet in triplets]
            for category, triplets in sorted(_ANGLE_DEF_CATEGORIES.items())
        },
        "scoring_constants": SCORING_CONSTANTS,
    }


def build_bundle(reference_dir: Path, thresholds_path: Path | None) -> tuple[dict, list[np.ndarray]]:
    """Build the manifest and the ordered list of arrays to concatenate.

    Returns ``(manifest, arrays)``. The manifest's per-reference ``offset`` values
    are byte offsets into the concatenation of ``arrays`` in the order returned.
    """
    references = load_reference_pose_library(str(reference_dir))
    if not references:
        raise SystemExit(f"no references loaded from {reference_dir}")

    thresholds = _load_label_thresholds(str(thresholds_path)) if thresholds_path else {}

    arrays: list[np.ndarray] = []
    byte_offset = 0
    techniques: dict[str, dict] = {}

    for technique in sorted(references):
        bank = references[technique]
        entries: dict[str, dict] = {}
        for angle in sorted(bank):
            seq = np.ascontiguousarray(bank[angle], dtype="<f4")
            if seq.ndim != 3 or seq.shape[1] != 17 or seq.shape[2] != 3:
                print(f"warning: skipping {technique}/{angle} with unexpected shape {seq.shape}")
                continue
            nbytes = int(seq.nbytes)
            entries[angle] = {
                "frames": int(seq.shape[0]),
                "offset": byte_offset,
                "byte_length": nbytes,
            }
            arrays.append(seq)
            byte_offset += nbytes

        if not entries:
            print(f"warning: technique '{technique}' had no usable references; omitted")
            continue

        techniques[technique] = {
            "angle_category": _technique_angle_category(technique),
            "score_threshold": thresholds.get(_normalize_key(technique)),
            "references": entries,
        }

    manifest = _shared_manifest_fields()
    manifest["dtype"] = "float32"
    manifest["byte_order"] = "little"
    manifest["shape_per_frame"] = [17, 3]
    manifest["total_bytes"] = byte_offset
    manifest["technique_count"] = len(techniques)
    manifest["reference_count"] = sum(len(t["references"]) for t in techniques.values())
    manifest["techniques"] = techniques
    return manifest, arrays


def build_json_bundle(reference_dir: Path, thresholds_path: Path | None) -> dict:
    """Self-contained JSON variant with coordinates inlined instead of a blob.

    Larger and slower to parse than the binary form -- meant for inspecting the
    data by hand, not for shipping.
    """
    manifest, _ = build_bundle(reference_dir, thresholds_path)
    references = load_reference_pose_library(str(reference_dir))

    for technique, meta in manifest["techniques"].items():
        for angle, entry in meta["references"].items():
            seq = np.asarray(references[technique][angle], dtype=np.float32)
            entry.pop("offset", None)
            entry.pop("byte_length", None)
            entry["data"] = seq.tolist()

    for key in ("dtype", "byte_order", "total_bytes"):
        manifest.pop(key, None)
    manifest["inline_data"] = True
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--reference-dir", default="reference_poses", help="reference pose root (default: %(default)s)"
    )
    parser.add_argument(
        "--thresholds",
        default="reference_poses/label_thresholds.json",
        help="per-label score thresholds; skipped without complaint if absent (default: %(default)s)",
    )
    parser.add_argument("--out-dir", default="export/mobile", help="output directory (default: %(default)s)")
    parser.add_argument(
        "--format",
        choices=("binary", "json"),
        default="binary",
        help="binary = manifest + .bin blob (ship this); json = coordinates inlined (default: %(default)s)",
    )
    args = parser.parse_args()

    reference_dir = _resolve_project_path(args.reference_dir)
    thresholds_path = _resolve_project_path(args.thresholds)
    out_dir = _resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not thresholds_path.exists():
        print(f"note: no thresholds at {thresholds_path}; score_threshold will be null for every technique")
        thresholds_path = None

    if args.format == "json":
        bundle = build_json_bundle(reference_dir, thresholds_path)
        out_json = out_dir / "reference_bundle.json"
        out_json.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        print(f"wrote {out_json} ({out_json.stat().st_size / 1e6:.2f} MB, coordinates inlined)")
        print(f"  techniques: {bundle['technique_count']}, references: {bundle['reference_count']}")
        return

    manifest, arrays = build_bundle(reference_dir, thresholds_path)

    blob = b"".join(arr.tobytes(order="C") for arr in arrays)
    if len(blob) != manifest["total_bytes"]:
        raise SystemExit(
            f"blob length {len(blob)} does not match manifest total_bytes {manifest['total_bytes']}"
        )

    out_bin = out_dir / "reference_bundle.bin"
    out_bin.write_bytes(blob)

    manifest["data_file"] = out_bin.name
    manifest["data_sha256"] = hashlib.sha256(blob).hexdigest()

    out_json = out_dir / "reference_bundle.json"
    out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote {out_bin} ({len(blob) / 1e6:.2f} MB)")
    print(f"wrote {out_json} ({out_json.stat().st_size / 1e3:.1f} kB manifest)")
    print(f"  techniques: {manifest['technique_count']}, references: {manifest['reference_count']}")
    print(f"  sha256: {manifest['data_sha256']}")


if __name__ == "__main__":
    main()
