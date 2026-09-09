"""Stage 1: run the pose model over each video once and cache what it saw.

This is the expensive half of reference capture — decode plus YOLO tracking —
separated out so it happens once per video instead of once per saved example.
The old flow launched one ``action_recognition.py`` subprocess per ``.npy``
file: 208 launches for the 52 ready plan rows, each paying interpreter start,
``import torch``, CUDA context creation, model load and cuDNN warmup before
decoding a single frame, and each re-inferring videos that other rows had
already processed.

Here one process loads the model once and walks a list of videos. The output is
a detection table per video under ``cache/tracks/``, which
``select_reference_windows.py`` replays to choose reference windows with no GPU
and no network — so re-tuning a capture gate no longer costs a re-run of this
stage.

Nothing about window selection happens here. No annotator, no video writer, no
reference library, no trainer scoring: just decode, track, record.

Usage::

    python scripts/extract_tracks.py --from-plan            # every video the plan names
    python scripts/extract_tracks.py --videos a.mp4 b.mp4
    python scripts/extract_tracks.py --from-plan --technique jab
    python scripts/extract_tracks.py --from-plan --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import reference_cache as rc  # noqa: E402

# Matches action_recognition.run(): the value it passes to extract_pose_instances.
POSE_CONF_THRES = 0.01
YOLO_CLASSES = (0,)
DEFAULT_TRACKER = "botsort.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and cache pose detections for videos, one model load for all of them."
    )
    source = parser.add_argument_group("sources")
    source.add_argument("--videos", nargs="*", default=[], help="explicit video paths or URLs")
    source.add_argument(
        "--from-plan",
        action="store_true",
        help="extract every distinct source the capture plan names",
    )
    source.add_argument(
        "--plan",
        default="reference_poses/generated_capture_plan_all_labels.csv",
        help="capture plan CSV used by --from-plan",
    )
    source.add_argument("--technique", default=None, help="with --from-plan, limit to one technique")

    parser.add_argument("--weights", default="yolo26n-pose.pt", help="pose model weights")
    parser.add_argument("--device", default="", help="cuda / cpu / mps (default: auto)")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size (default: 640)")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="half precision on CUDA. Changes keypoints slightly, so it is part of the cache key",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1800,
        help=(
            "stop after this many frames per video (default: 1800, matching the capture "
            "runs' --reference-search-max-frames). 0 extracts the whole video"
        ),
    )
    parser.add_argument("--force", action="store_true", help="re-extract videos that are already cached")
    parser.add_argument(
        "--no-video-cache",
        dest="video_cache",
        action="store_false",
        help="stream URLs instead of downloading them into cache/videos/ first",
    )
    parser.add_argument("--dry-run", action="store_true", help="list what would be extracted, then exit")
    return parser.parse_args()


def _reset_tracker(model) -> None:
    """Clear tracker state between videos.

    ``model.track(..., persist=True)`` is required within a video — it is what
    keeps track ids stable across frames — but it also carries state *across*
    videos. Without this reset the second video inherits the first one's tracks
    and its ids continue counting upward, so a cached table would depend on
    which videos happened to be extracted before it in the same process. Both
    the per-tracker state and the global id counter have to go.
    """
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", None) or []:
        reset = getattr(tracker, "reset", None)
        if callable(reset):
            reset()
    try:
        from ultralytics.trackers.basetrack import BaseTrack

        BaseTrack.reset_id()
    except Exception as exc:  # pragma: no cover - depends on ultralytics internals
        print(f"  warning: could not reset the global track id counter ({exc})")


def extract_video(
    model,
    video_path: str,
    *,
    imgsz: int,
    fp16: bool,
    max_frames: int,
    source_label: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]] | None:
    """Decode one video, track through it, and return the detection tables.

    The frame loop mirrors ``action_recognition.run()``: same 1-based frame
    counter, same ``track(..., persist=True, classes=[0])`` call, same
    ``extract_pose_instances`` threshold. Boxes and poses are collected into
    separate tables because the pose list can be shorter than the box list.
    """
    import cv2

    from action_recognition import extract_pose_instances

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  failed to open: {video_path}")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    frame_idx: list[int] = []
    det_order: list[int] = []
    track_ids: list[int] = []
    boxes: list[np.ndarray] = []
    pose_frame_idx: list[int] = []
    pose_order: list[int] = []
    pose_kpts: list[np.ndarray] = []

    frame_counter = 0
    reached_eof = False
    started = time.perf_counter()

    _reset_tracker(model)
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                reached_eof = True
                break

            frame_counter += 1
            if max_frames > 0 and frame_counter > max_frames:
                frame_counter -= 1
                break

            results = model.track(
                frame, persist=True, classes=list(YOLO_CLASSES), verbose=False, imgsz=imgsz, half=fp16
            )
            if not results or not results[0].boxes.is_track:
                continue

            frame_boxes = results[0].boxes.xyxy.cpu().numpy()
            frame_track_ids = results[0].boxes.id.cpu().numpy()
            for order, (box, tid) in enumerate(zip(frame_boxes, frame_track_ids)):
                frame_idx.append(frame_counter)
                det_order.append(order)
                track_ids.append(int(tid))
                boxes.append(np.asarray(box, dtype=np.float32))

            for order, pose in enumerate(extract_pose_instances(results[0], conf_thres=POSE_CONF_THRES)):
                pose_frame_idx.append(frame_counter)
                pose_order.append(order)
                pose_kpts.append(np.asarray(pose, dtype=np.float32))
    finally:
        cap.release()

    if not pose_kpts:
        print(f"  no pose detections in {frame_counter} frame(s) — not caching")
        return None

    shapes = {p.shape for p in pose_kpts}
    if len(shapes) != 1:
        # A single (N,K,3) array cannot hold mixed keypoint counts. Rather than
        # cache something the replay would have to second-guess, skip the video
        # and let it fall back to the live capture path.
        print(f"  inconsistent keypoint shapes {shapes} — not caching")
        return None

    elapsed = time.perf_counter() - started
    tables = {
        "frame_idx": np.array(frame_idx, dtype=np.int32),
        "det_order": np.array(det_order, dtype=np.int16),
        "track_id": np.array(track_ids, dtype=np.int32),
        "box": np.stack(boxes).astype(np.float32) if boxes else np.zeros((0, 4), np.float32),
        "pose_frame_idx": np.array(pose_frame_idx, dtype=np.int32),
        "pose_order": np.array(pose_order, dtype=np.int16),
        "pose_kpts": np.stack(pose_kpts).astype(np.float32),
    }
    meta: dict[str, object] = {
        "source": source_label,
        "local_path": str(video_path),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "fps": fps,
        "frames_read": frame_counter,
        "reached_eof": reached_eof,
        "max_frames": int(max_frames),
        "imgsz": int(imgsz),
        "fp16": bool(fp16),
        "conf_thres": POSE_CONF_THRES,
        "kpt_shape": list(next(iter(shapes))),
        "n_boxes": len(frame_idx),
        "n_poses": len(pose_frame_idx),
        "distinct_tracks": len(set(track_ids)),
        "extract_seconds": round(elapsed, 2),
    }
    return tables, meta


def _plan_sources(plan_path: Path, technique: str | None) -> dict[str, str]:
    rows = rc.plan_rows(plan_path)
    if technique:
        wanted = technique.strip().lower()
        rows = [r for r in rows if (r.get("technique") or "").strip().lower() == wanted]
    by_id: dict[str, str] = {}
    for row in rows:
        for url in rc.collect_source_urls(row):
            by_id.setdefault(rc.video_id_for_source(url), url)
    return by_id


def main() -> int:
    args = parse_args()

    targets: dict[str, str] = {}
    if args.from_plan:
        plan_path = Path(args.plan)
        if not plan_path.is_absolute():
            plan_path = PROJECT_ROOT / plan_path
        if not plan_path.exists():
            print(f"plan file not found: {plan_path}")
            return 2
        targets.update(_plan_sources(plan_path, args.technique))
    for video in args.videos:
        targets.setdefault(rc.video_id_for_source(video), video)

    if not targets:
        print("no sources given (use --from-plan or --videos)")
        return 2

    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path

    # Resolve the device before hashing: a CUDA table and a CPU table are not
    # interchangeable, so the key has to know which one produced it.
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import select_device
    import ultralytics

    device = select_device(args.device)
    device_kind = getattr(device, "type", str(device))
    signature = rc.extraction_signature(
        weights=weights_path,
        imgsz=args.imgsz,
        fp16=args.fp16,
        conf_thres=POSE_CONF_THRES,
        classes=YOLO_CLASSES,
        tracker=DEFAULT_TRACKER,
        ultralytics_version=ultralytics.__version__,
        device_kind=str(device_kind),
    )
    print(f"extraction signature: {signature}  (device={device_kind}, imgsz={args.imgsz}, fp16={args.fp16})")

    pending: list[tuple[str, str]] = []
    for video_id, source in targets.items():
        if not args.force:
            cached = rc.read_track_cache(video_id, signature)
            if cached is not None and rc.cache_covers_frames(cached, args.max_frames):
                continue
        pending.append((video_id, source))

    print(f"{len(targets)} distinct source(s), {len(pending)} to extract")
    if args.dry_run:
        for video_id, source in pending:
            print(f"  would extract {video_id}  {source}")
        return 0
    if not pending:
        print("nothing to do — every source is already cached")
        return 0

    fp16 = bool(args.fp16) and str(device_kind) == "cuda"
    if args.fp16 and not fp16:
        print("note: --fp16 ignored off CUDA")

    model = YOLO(str(weights_path)).to(device)
    if getattr(model, "task", "") != "pose":
        print(f"weights are not a pose model (task={getattr(model, 'task', '?')})")
        return 2

    extracted = 0
    failed: list[str] = []
    for i, (video_id, source) in enumerate(pending, start=1):
        print(f"[{i}/{len(pending)}] {video_id}  {source}")

        local = source
        if rc.is_url(source):
            if args.video_cache:
                cached_video = rc.ensure_local_video(source, quiet=True)
                if cached_video is not None:
                    local = str(cached_video)
                else:
                    print("  download failed; falling back to streaming")
                    local = _stream_url(source)
            else:
                local = _stream_url(source)

        result = extract_video(
            model,
            local,
            imgsz=args.imgsz,
            fp16=fp16,
            max_frames=args.max_frames,
            source_label=source,
        )
        if result is None:
            failed.append(source)
            continue

        tables, meta = result
        out_path = rc.write_track_cache(video_id, signature, meta=meta, **tables)
        print(
            f"  cached {meta['n_poses']} pose(s) over {meta['frames_read']} frame(s), "
            f"{meta['distinct_tracks']} track(s) -> {out_path.name} "
            f"({out_path.stat().st_size / 1024:.0f} KB, {meta['extract_seconds']}s)"
        )
        extracted += 1

    print("summary:", {"extracted": extracted, "failed": len(failed), "distinct_sources": len(targets)})
    if failed:
        print("failed sources:")
        for source in failed:
            print(f"  {source}")
    return 0 if not failed else 1


def _stream_url(source: str) -> str:
    """Resolve a YouTube page URL to a playable stream, as run() does."""
    try:
        from ultralytics.data.loaders import get_best_youtube_url

        return get_best_youtube_url(source)
    except Exception as exc:
        print(f"  warning: could not resolve stream ({exc}), using the URL as-is")
        return source


if __name__ == "__main__":
    raise SystemExit(main())
