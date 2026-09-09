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
    device_kind: str,
) -> str:
    """Hash of everything that changes the detections stored in a track cache.

    Selection-only parameters are deliberately absent — see the module
    docstring. ``ultralytics_version`` is included because tracker behaviour is
    version-sensitive, and ``device_kind`` because CUDA and CPU float paths do
    not produce identical keypoints; both are passed in so this module stays
    importable without ultralytics or torch installed.

    How many frames were extracted is deliberately *not* part of the key.
    Folding it in would give the same video a different entry for every search
    horizon, fragmenting the cache exactly where it is meant to pay off. The
    count is recorded in the sidecar instead and compared by
    ``cache_covers_frames`` at lookup time.
    """
    payload = {
        "weights": weights_fingerprint(weights),
        "imgsz": int(imgsz),
        "fp16": bool(fp16),
        "conf_thres": round(float(conf_thres), 6),
        "classes": sorted(int(c) for c in classes),
        "tracker": str(tracker),
        "ultralytics": str(ultralytics_version),
        "device_kind": str(device_kind),
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
    det_order: np.ndarray,
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
        det_order=np.asarray(det_order, dtype=np.int16),
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
        det_order: np.ndarray,
        track_id: np.ndarray,
        box: np.ndarray,
        kpts: np.ndarray,
        meta: dict[str, Any],
    ) -> None:
        self.frame_idx = np.asarray(frame_idx, dtype=np.int32)
        self.det_order = np.asarray(det_order, dtype=np.int16)
        self.track_id = np.asarray(track_id, dtype=np.int32)
        self.box = np.asarray(box, dtype=np.float32)
        self.kpts = np.asarray(kpts, dtype=np.float32)
        self.meta = dict(meta)

        # Sort by (frame, detection order). Detection order is the model's own
        # output order and is load-bearing: run() pairs keypoints to boxes
        # positionally, and _select_primary_track breaks score ties with max(),
        # which returns the first key at the maximum. Reordering rows would
        # reassign keypoints to the wrong person and flip tie-breaks.
        order = np.lexsort((self.det_order, self.frame_idx))
        if not np.array_equal(order, np.arange(self.frame_idx.size)):
            self.frame_idx = self.frame_idx[order]
            self.det_order = self.det_order[order]
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
            det_order = data["det_order"]
            track_id = data["track_id"]
            box = data["box"]
            kpts = data["kpts"]
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"warning: ignoring unreadable track cache {npz_path.name}: {exc}")
        return None
    return TrackCache(frame_idx, det_order, track_id, box, kpts, meta)


def cache_covers_frames(cache: "TrackCache", needed_frames: int) -> bool:
    """Is a cached extraction usable for a run that wants ``needed_frames``?

    ``needed_frames`` of 0 means "the whole video". A cache that stopped early
    because it hit its own frame budget covers a smaller request (the replay
    just stops sooner) but not a larger one. A cache that ran to end-of-file
    covers any request, because there was nothing more to extract.
    """
    if bool(cache.meta.get("reached_eof", False)):
        return True
    extracted = int(cache.meta.get("frames_read", 0))
    if needed_frames <= 0:
        return False
    return extracted >= int(needed_frames)


# ---------------------------------------------------------------------------
# Video cache
# ---------------------------------------------------------------------------

def _run_yt_dlp(url: str, out_template: str, fmt: str, extra_args: list[str] | None = None) -> tuple[bool, str]:
    import subprocess

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--no-progress",
        "--quiet",
        "--no-warnings",
        "-f",
        fmt,
        "--merge-output-format",
        "mp4",
        "-o",
        out_template,
        url,
    ]
    if extra_args:
        cmd.extend(extra_args)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except FileNotFoundError:
        return False, "yt-dlp is not installed (pip install yt-dlp)"
    except subprocess.TimeoutExpired:
        return False, "yt-dlp timed out after 1800s"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        return False, detail[-1] if detail else f"yt-dlp exited {result.returncode}"
    return True, ""


DEFAULT_FORMAT = "bestvideo[height<=1080][ext=mp4]+bestaudio/best[height<=1080]/best"


def ensure_local_video(
    source: str,
    *,
    fmt: str = DEFAULT_FORMAT,
    force: bool = False,
    quiet: bool = False,
) -> Path | None:
    """Return a local file for ``source``, downloading it once if necessary.

    Local paths are returned as-is. URLs are fetched into ``cache/videos/``
    under their stable ``video_id``, so the 60 plan URLs that appear in more
    than one row are downloaded once rather than re-streamed per use.

    Returns None when the source cannot be made local (no yt-dlp, download
    failure). Callers are expected to fall back to streaming rather than fail:
    a missing cache should slow the pipeline down, not break it.
    """
    text = str(source).strip()
    if not is_url(text):
        path = Path(text)
        return path if path.exists() else None

    video_id = video_id_for_source(text)
    if not force:
        cached = find_cached_video(video_id)
        if cached is not None:
            return cached

    VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Download to a temp stem and rename on success, so an interrupted fetch
    # never leaves a truncated file that later runs would treat as a cache hit.
    tmp_template = str(VIDEO_CACHE_DIR / f".{video_id}.partial.%(ext)s")
    for stale in VIDEO_CACHE_DIR.glob(f".{video_id}.partial.*"):
        stale.unlink(missing_ok=True)

    ok, error = _run_yt_dlp(text, tmp_template, fmt)
    if not ok:
        if not quiet:
            print(f"  download failed for {text}: {error}")
        for stale in VIDEO_CACHE_DIR.glob(f".{video_id}.partial.*"):
            stale.unlink(missing_ok=True)
        return None

    produced = sorted(VIDEO_CACHE_DIR.glob(f".{video_id}.partial.*"))
    if not produced:
        if not quiet:
            print(f"  download produced no file for {text}")
        return None

    downloaded = produced[0]
    final_path = VIDEO_CACHE_DIR / f"{video_id}{downloaded.suffix}"
    downloaded.replace(final_path)
    for stale in produced[1:]:
        stale.unlink(missing_ok=True)

    (VIDEO_CACHE_DIR / f"{video_id}.json").write_text(
        json.dumps(
            {
                "video_id": video_id,
                "source_url": text,
                "format_selector": fmt,
                "file": final_path.name,
                "size_bytes": final_path.stat().st_size,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return final_path


# ---------------------------------------------------------------------------
# Plan CSV source columns
# ---------------------------------------------------------------------------

def parse_source_urls(raw_source: str) -> list[str]:
    """Split one CSV cell into candidate URLs, order-preserving and deduped.

    Supported separators: newline, comma, semicolon, pipe.
    """
    if not raw_source:
        return []
    parts = [p.strip() for p in re.split(r"[\n,;|]+", raw_source) if p.strip()]
    seen: set[str] = set()
    unique: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def collect_source_urls(row: dict[str, str]) -> list[str]:
    """Distinct source URLs for one plan row, in column order.

    Reads ``source_url_1``..``source_url_4`` plus the legacy single
    ``source_url`` column.
    """
    urls: list[str] = []
    for i in range(1, 5):
        cell = (row.get(f"source_url_{i}") or "").strip()
        if cell:
            urls.extend(parse_source_urls(cell))
    legacy = (row.get("source_url") or "").strip()
    if legacy:
        urls.extend(parse_source_urls(legacy))

    seen: set[str] = set()
    unique: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def plan_rows(plan_path: Path, ready_only: bool = True) -> list[dict[str, str]]:
    """Read a capture plan CSV, optionally keeping only ``command_ready=yes`` rows.

    Each returned row carries ``_csv_line`` for error messages that point at the
    spreadsheet the user actually edits.
    """
    import csv

    rows: list[dict[str, str]] = []
    with plan_path.open(encoding="utf-8-sig") as handle:
        for csv_line, row in enumerate(csv.DictReader(handle), start=2):  # header is line 1
            if ready_only and (row.get("command_ready", "").strip().lower() != "yes"):
                continue
            row_with_meta = dict(row)
            row_with_meta["_csv_line"] = str(csv_line)
            rows.append(row_with_meta)
    return rows
