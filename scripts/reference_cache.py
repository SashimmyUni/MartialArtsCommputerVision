"""Shared cache layout and keys for the reference-collection pipeline.

Reference collection used to re-download and re-infer a video every time it was
needed. The 52 ready rows of ``generated_capture_plan_all_labels.csv`` ask for
208 saved examples but name only 119 distinct URLs, so most videos were fetched
and run through the pose model several times over, and re-tuning a capture gate
meant paying for all of it again.

This module defines the two caches that fix that:

``cache/videos/<video_id>.<ext>``
    The source video, downloaded once (see ``prefetch_sources.py``).

``cache/tracks/<video_id>__<signature>.npz``
    Every pose detection the model produced for that video (see
    ``extract_tracks.py``), which ``select_reference_windows.py`` replays to
    pick reference windows without touching the GPU or the network.

The signature is the important part. It covers everything that changes the
*detections* — weights, image size, precision, tracker, ultralytics version —
and deliberately nothing that only changes *which window gets selected*
(``skip_frame``, the ``ref_*`` gates, ``num_video_sequence_samples``,
``person_selection_mode``). That asymmetry is what makes gate re-tuning free:
change a threshold, replay the same cached tracks.

Kept free of torch/cv2/ultralytics imports so the selection stage and its tests
can use it without the heavy stack.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CACHE_ROOT = PROJECT_ROOT / "cache"
VIDEO_CACHE_DIR = CACHE_ROOT / "videos"
TRACK_CACHE_DIR = CACHE_ROOT / "tracks"

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v")

_YOUTUBE_HOSTS = {"www.youtube.com", "youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}
_YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def is_url(source: str) -> bool:
    return str(source).startswith(("http://", "https://"))


def youtube_id(url: str) -> str | None:
    """Return the 11-character YouTube video id in ``url``, or None.

    Handles the ``watch?v=``, ``youtu.be/``, ``/embed/`` and ``/shorts/`` forms,
    with arbitrary extra query parameters.
    """
    try:
        parsed = urlparse(str(url))
    except ValueError:
        return None
    if parsed.hostname not in _YOUTUBE_HOSTS:
        return None

    candidates: list[str] = []
    query_v = parse_qs(parsed.query).get("v")
    if query_v:
        candidates.append(query_v[0])

    parts = [p for p in parsed.path.split("/") if p]
    if parts:
        if parsed.hostname in {"youtu.be", "www.youtu.be"}:
            candidates.append(parts[0])
        elif parts[0] in {"embed", "shorts", "live", "v"} and len(parts) > 1:
            candidates.append(parts[1])

    for candidate in candidates:
        if _YOUTUBE_ID_RE.match(candidate):
            return candidate
    return None


def _short_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def video_id_for_source(source: str) -> str:
    """Stable cache id for a plan source URL or a local video path.

    YouTube URLs key on their video id, so the same video reached through
    different URL spellings (``youtu.be`` vs ``watch?v=``, extra tracking
    parameters) resolves to one cache entry.

    A file already living in ``cache/videos/`` keys on its own stem, so that a
    prefetched video hands back the id it was fetched under rather than a new
    content hash — prefetch and extract have to agree on the id, or the track
    cache is written under a key nothing looks up.
    """
    text = str(source).strip()

    vid = youtube_id(text)
    if vid:
        return vid

    if is_url(text):
        return f"url_{_short_hash(text)}"

    path = Path(text)
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path

    try:
        if resolved.parent == VIDEO_CACHE_DIR.resolve():
            return resolved.stem
    except OSError:
        pass

    if resolved.exists():
        stat = resolved.stat()
        fingerprint = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"
    else:
        fingerprint = str(resolved)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", resolved.stem)[:40] or "clip"
    return f"local_{slug}_{_short_hash(fingerprint, 12)}"


def weights_fingerprint(weights_path: str | Path) -> str:
    """Identify the weights file by content when readable, else by name+size.

    A weights swap must invalidate the track cache; two checkouts of the same
    weights must not.
    """
    path = Path(weights_path)
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return f"{path.name}:{digest.hexdigest()[:16]}"
    except OSError:
        return f"{path.name}:unreadable"


def extraction_signature(
    *,
    weights: str | Path,
    imgsz: int,
    fp16: bool,
    conf_thres: float,
    classes: tuple[int, ...] | list[int],
    tracker: str,
    ultralytics_version: str,
    max_frames: int,
    detect_stride: int = 1,
) -> str:
    """Hash of everything that changes the detections stored in a track cache.

    Selection-only parameters are deliberately absent — see the module
    docstring. ``ultralytics_version`` is included because tracker behaviour is
    version-sensitive, and the caller passes it in so this module stays
    importable without ultralytics installed.
    """
    payload = {
        "weights": weights_fingerprint(weights),
        "imgsz": int(imgsz),
        "fp16": bool(fp16),
        "conf_thres": round(float(conf_thres), 6),
        "classes": sorted(int(c) for c in classes),
        "tracker": str(tracker),
        "ultralytics": str(ultralytics_version),
        "max_frames": int(max_frames),
        "detect_stride": int(detect_stride),
    }
    return _short_hash(json.dumps(payload, sort_keys=True), 16)


def track_cache_path(video_id: str, signature: str) -> Path:
    return TRACK_CACHE_DIR / f"{video_id}__{signature}.npz"


def track_meta_path(video_id: str, signature: str) -> Path:
    return TRACK_CACHE_DIR / f"{video_id}__{signature}.json"


def find_cached_video(video_id: str) -> Path | None:
    """Return the prefetched video file for ``video_id``, if one exists."""
    for ext in VIDEO_EXTENSIONS:
        candidate = VIDEO_CACHE_DIR / f"{video_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def write_track_cache(
    video_id: str,
    signature: str,
    *,
    frame_idx: np.ndarray,
    track_id: np.ndarray,
    box: np.ndarray,
    kpts: np.ndarray,
    meta: dict[str, Any],
) -> Path:
    """Persist one video's detections as a flat event table.

    Flat rather than per-frame because detection counts are ragged. Row order
    within a frame is the model's own output order and must be preserved:
    ``action_recognition.run()`` pairs keypoints to boxes positionally, so
    reordering rows would silently re-assign keypoints to the wrong person.
    """
    TRACK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = track_cache_path(video_id, signature)
    np.savez_compressed(
        out_path,
        frame_idx=np.asarray(frame_idx, dtype=np.int32),
        track_id=np.asarray(track_id, dtype=np.int32),
        box=np.asarray(box, dtype=np.float32),
        kpts=np.asarray(kpts, dtype=np.float32),
    )
    meta_out = dict(meta)
    meta_out["video_id"] = video_id
    meta_out["signature"] = signature
    track_meta_path(video_id, signature).write_text(
        json.dumps(meta_out, indent=2, sort_keys=True), encoding="utf-8"
    )
    return out_path


class TrackCache:
    """A loaded track cache, indexed by frame.

    ``frames_read`` is carried separately from the event table because frames
    where the model tracked nobody produce no rows but still advance the frame
    counter. The replay has to keep those gaps: renumbering frames would shift
    every cooldown and window span computed from them.
    """

    def __init__(
        self,
        frame_idx: np.ndarray,
        track_id: np.ndarray,
        box: np.ndarray,
        kpts: np.ndarray,
        meta: dict[str, Any],
    ) -> None:
        self.frame_idx = np.asarray(frame_idx, dtype=np.int32)
        self.track_id = np.asarray(track_id, dtype=np.int32)
        self.box = np.asarray(box, dtype=np.float32)
        self.kpts = np.asarray(kpts, dtype=np.float32)
        self.meta = dict(meta)

        order = np.argsort(self.frame_idx, kind="stable")
        if not np.array_equal(order, np.arange(self.frame_idx.size)):
            self.frame_idx = self.frame_idx[order]
            self.track_id = self.track_id[order]
            self.box = self.box[order]
            self.kpts = self.kpts[order]

        # Row span per frame, so the replay can slice instead of scanning.
        self._starts: dict[int, tuple[int, int]] = {}
        if self.frame_idx.size:
            boundaries = np.flatnonzero(np.diff(self.frame_idx)) + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [self.frame_idx.size]))
            for s, e in zip(starts, ends):
                self._starts[int(self.frame_idx[s])] = (int(s), int(e))

    @property
    def frame_width(self) -> int:
        return int(self.meta.get("frame_width", 0))

    @property
    def frame_height(self) -> int:
        return int(self.meta.get("frame_height", 0))

    @property
    def frames_read(self) -> int:
        return int(self.meta.get("frames_read", int(self.frame_idx.max()) if self.frame_idx.size else 0))

    @property
    def source(self) -> str:
        return str(self.meta.get("source", ""))

    def rows_for_frame(self, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(track_ids, boxes, kpts)`` for one 1-based frame number."""
        span = self._starts.get(int(frame))
        if span is None:
            empty_kpts = self.kpts[:0]
            return self.track_id[:0], self.box[:0], empty_kpts
        s, e = span
        return self.track_id[s:e], self.box[s:e], self.kpts[s:e]


def read_track_cache(video_id: str, signature: str) -> TrackCache | None:
    """Load a cached track table, or None when it is absent or unreadable."""
    npz_path = track_cache_path(video_id, signature)
    meta_path = track_meta_path(video_id, signature)
    if not npz_path.exists() or not meta_path.exists():
        return None
    try:
        with np.load(npz_path) as data:
            frame_idx = data["frame_idx"]
            track_id = data["track_id"]
            box = data["box"]
            kpts = data["kpts"]
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"warning: ignoring unreadable track cache {npz_path.name}: {exc}")
        return None
    return TrackCache(frame_idx, track_id, box, kpts, meta)
