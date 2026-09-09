"""Check the cache keys and the track-table round trip in scripts/reference_cache.py.

The cache is only useful if its key draws the line in exactly the right place:

- Anything that changes the *detections* — weights, image size, precision,
  tracker, ultralytics version, device — must produce a different key, or a
  stale table gets replayed as if it were current.
- Anything that only changes *which window is selected* — the ``ref_*`` gates,
  ``skip_frame``, window length — must **not**, because re-tuning those against
  cached tracks without re-running the model is the entire point.

Also checks that the detection table survives a save/load round trip with row
order intact. Order is load-bearing: ``run()`` pairs keypoints to boxes
positionally, and ``_select_primary_track`` breaks ties with ``max()``, which
returns the first key at the maximum.

Needs only numpy. Run from anywhere:

    python test_reference_cache.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "scripts"))

import reference_cache as rc  # noqa: E402

BASE_SIGNATURE_ARGS = dict(
    weights="yolo26n-pose.pt",
    imgsz=640,
    fp16=False,
    conf_thres=0.01,
    classes=[0],
    tracker="botsort.yaml",
    ultralytics_version="8.4.21",
    device_kind="cuda",
)


def test_youtube_id_canonicalisation() -> None:
    """One video reached by different URL spellings must be one cache entry."""
    same = [
        "https://www.youtube.com/watch?v=SjLXzCpRS8U",
        "https://youtu.be/SjLXzCpRS8U",
        "https://youtu.be/SjLXzCpRS8U?t=42",
        "https://www.youtube.com/watch?v=SjLXzCpRS8U&list=PLxyz&index=3",
        "https://m.youtube.com/watch?v=SjLXzCpRS8U",
        "https://www.youtube.com/shorts/SjLXzCpRS8U",
        "https://www.youtube.com/embed/SjLXzCpRS8U",
    ]
    ids = {rc.video_id_for_source(url) for url in same}
    assert ids == {"SjLXzCpRS8U"}, f"expected one id, got {ids}"

    assert rc.youtube_id("https://vimeo.com/12345") is None
    assert rc.youtube_id("https://www.youtube.com/watch?v=too_short") is None
    # A non-YouTube URL still gets a stable id rather than colliding with others.
    a = rc.video_id_for_source("https://example.com/a.mp4")
    b = rc.video_id_for_source("https://example.com/b.mp4")
    assert a != b and a.startswith("url_")
    print("OK: youtube id canonicalisation")


def test_signature_covers_detection_parameters() -> None:
    base = rc.extraction_signature(**BASE_SIGNATURE_ARGS)
    assert rc.extraction_signature(**BASE_SIGNATURE_ARGS) == base, "signature must be deterministic"

    must_change = {
        "imgsz": 960,
        "fp16": True,
        "conf_thres": 0.25,
        "classes": [0, 1],
        "tracker": "bytetrack.yaml",
        "ultralytics_version": "8.5.0",
        "device_kind": "cpu",
    }
    for field, value in must_change.items():
        altered = rc.extraction_signature(**{**BASE_SIGNATURE_ARGS, field: value})
        assert altered != base, f"changing {field} must invalidate the track cache"
    print(f"OK: {len(must_change)} detection parameter(s) invalidate the cache")


def test_signature_ignores_selection_parameters() -> None:
    """Gate re-tuning must hit the cache, not rebuild it.

    ``extraction_signature`` takes no selection parameters at all, so this is
    really a guard against someone later adding one to the payload: if a
    ``ref_*`` threshold or window length ever reaches this function, re-tuning
    silently starts costing a full re-inference again.
    """
    import inspect

    params = set(inspect.signature(rc.extraction_signature).parameters)
    selection_only = {
        "skip_frame",
        "num_video_sequence_samples",
        "ref_min_motion_energy",
        "ref_min_return_closure",
        "ref_min_score_gate",
        "ref_stance_start_threshold",
        "ref_stance_end_threshold",
        "ref_stance_peak_threshold",
        "ref_stance_min_frames",
        "ref_stance_hold_frames",
        "person_selection_mode",
        "reference_sequence_mode",
        "max_frames",
    }
    leaked = params & selection_only
    assert not leaked, f"selection-only parameters must not be in the cache key: {sorted(leaked)}"
    print(f"OK: no selection parameter reaches the cache key ({len(params)} key inputs)")


def _synthetic_table(n_frames: int = 6, people: int = 2, drop_pose_on: int | None = None):
    """Build a small detection table.

    ``drop_pose_on`` omits the *first* pose instance on that frame while keeping
    both boxes, reproducing what ``extract_pose_instances`` does when an
    instance has no confident keypoint.
    """
    rng = np.random.default_rng(0)
    frame_idx, det_order, track_id, box = [], [], [], []
    pose_frame_idx, pose_order, pose_kpts = [], [], []
    for f in range(1, n_frames + 1):
        for d in range(people):
            frame_idx.append(f)
            det_order.append(d)
            track_id.append(d + 1)
            box.append([10.0 * d, 20.0 * d, 10.0 * d + 50, 20.0 * d + 90])
        kept = 0
        for d in range(people):
            if drop_pose_on == f and d == 0:
                continue
            pose_frame_idx.append(f)
            pose_order.append(kept)
            pose_kpts.append(rng.random((17, 3), dtype=np.float32))
            kept += 1
    return dict(
        frame_idx=np.array(frame_idx, dtype=np.int32),
        det_order=np.array(det_order, dtype=np.int16),
        track_id=np.array(track_id, dtype=np.int32),
        box=np.array(box, dtype=np.float32),
        pose_frame_idx=np.array(pose_frame_idx, dtype=np.int32),
        pose_order=np.array(pose_order, dtype=np.int16),
        pose_kpts=np.stack(pose_kpts).astype(np.float32),
    )


def _write_and_read(tables, meta):
    with tempfile.TemporaryDirectory() as tmp:
        original_dir = rc.TRACK_CACHE_DIR
        rc.TRACK_CACHE_DIR = Path(tmp)
        try:
            rc.write_track_cache("vid", "sig", meta=meta, **tables)
            return rc.read_track_cache("vid", "sig")
        finally:
            rc.TRACK_CACHE_DIR = original_dir


def test_track_cache_round_trip() -> None:
    tables = _synthetic_table()
    meta = {
        "frame_width": 1920,
        "frame_height": 1080,
        "fps": 30.0,
        "frames_read": 6,
        "reached_eof": True,
        "source": "synthetic",
    }
    loaded = _write_and_read(tables, meta)

    assert loaded is not None, "round trip lost the cache"
    assert np.array_equal(loaded.frame_idx, tables["frame_idx"])
    assert np.array_equal(loaded.det_order, tables["det_order"])
    assert np.array_equal(loaded.track_id, tables["track_id"])
    assert np.allclose(loaded.box, tables["box"])
    assert np.allclose(loaded.pose_kpts, tables["pose_kpts"])
    assert loaded.frame_width == 1920 and loaded.frame_height == 1080
    assert loaded.frames_read == 6

    ids, boxes = loaded.boxes_for_frame(3)
    assert list(ids) == [1, 2], f"expected both people on frame 3, got {ids}"
    assert boxes.shape == (2, 4)
    assert loaded.poses_for_frame(3).shape == (2, 17, 3)

    # A frame nobody was tracked on yields nothing rather than raising: those
    # gaps are real and the replay must keep them.
    ids, boxes = loaded.boxes_for_frame(99)
    assert ids.size == 0 and boxes.shape == (0, 4)
    assert loaded.poses_for_frame(99).shape[0] == 0
    print("OK: track cache round trip preserves rows, order and frame gaps")


def test_pose_table_may_be_shorter_than_box_table() -> None:
    """A filtered-out pose must stay filtered out.

    extract_pose_instances drops instances with no confident keypoint, but
    run() still pairs pose i to box i. Storing merged pairs would silently
    repair that mismatch and the replay would diverge from the live path.
    """
    tables = _synthetic_table(n_frames=4, people=2, drop_pose_on=2)
    loaded = _write_and_read(tables, {"frame_width": 640, "frame_height": 480, "frames_read": 4})
    assert loaded is not None

    ids, boxes = loaded.boxes_for_frame(2)
    poses = loaded.poses_for_frame(2)
    assert len(ids) == 2, "both boxes should survive"
    assert poses.shape[0] == 1, "one pose instance was dropped and must stay dropped"

    ids, _ = loaded.boxes_for_frame(3)
    assert loaded.poses_for_frame(3).shape[0] == len(ids) == 2
    print("OK: pose table stays independent of the box table")


def test_missing_cache_reads_as_none() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        original_dir = rc.TRACK_CACHE_DIR
        rc.TRACK_CACHE_DIR = Path(tmp)
        try:
            assert rc.read_track_cache("absent", "sig") is None
        finally:
            rc.TRACK_CACHE_DIR = original_dir
    print("OK: absent cache reads as a miss, not an error")


def test_frame_coverage() -> None:
    """A cache that stopped early covers a smaller request, not a larger one."""
    tables = _synthetic_table(n_frames=3)

    partial = rc.TrackCache(
        **tables,
        meta={"frame_width": 640, "frame_height": 480, "frames_read": 1800, "reached_eof": False},
    )
    assert rc.cache_covers_frames(partial, 1800)
    assert rc.cache_covers_frames(partial, 900)
    assert not rc.cache_covers_frames(partial, 3600), "must re-extract when more frames are wanted"
    assert not rc.cache_covers_frames(partial, 0), "'whole video' is not covered by a truncated cache"

    complete = rc.TrackCache(
        **tables,
        meta={"frame_width": 640, "frame_height": 480, "frames_read": 420, "reached_eof": True},
    )
    assert rc.cache_covers_frames(complete, 0), "a full extraction covers any request"
    assert rc.cache_covers_frames(complete, 100000)
    print("OK: frame coverage")


def test_plan_parsing() -> None:
    plan = rc.PROJECT_ROOT / "reference_poses" / "generated_capture_plan_all_labels.csv"
    if not plan.exists():
        print("SKIP: plan CSV not present")
        return

    rows = rc.plan_rows(plan)
    assert rows, "no command_ready rows found in the plan"

    urls = [u for row in rows for u in rc.collect_source_urls(row)]
    distinct = {rc.video_id_for_source(u) for u in urls}
    assert distinct, "plan rows carry no source URLs"
    assert len(distinct) <= len(urls)
    print(f"OK: plan parses to {len(rows)} ready row(s), {len(urls)} slots, {len(distinct)} distinct video(s)")

    multi = rc.parse_source_urls("a.mp4, b.mp4 | c.mp4\nd.mp4; a.mp4")
    assert multi == ["a.mp4", "b.mp4", "c.mp4", "d.mp4"], multi
    print("OK: multi-URL cell parsing dedupes and preserves order")


def main() -> None:
    test_youtube_id_canonicalisation()
    test_signature_covers_detection_parameters()
    test_signature_ignores_selection_parameters()
    test_track_cache_round_trip()
    test_pose_table_may_be_shorter_than_box_table()
    test_missing_cache_reads_as_none()
    test_frame_coverage()
    test_plan_parsing()
    print("\nall reference_cache checks passed")


if __name__ == "__main__":
    main()
