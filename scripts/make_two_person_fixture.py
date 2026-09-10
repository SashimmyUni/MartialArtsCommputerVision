"""Synthesize a two-person test clip for headless fight_analysis.py smoke tests.

The repo ships no fight footage (InputVideo/ is gitignored and absent), so
there is nothing to point fight_analysis.py at without a real video. This
mirrors the single-person fixture trick in
``.claude/skills/run-martial-arts-cv/driver.py``: crop person(s) out of
ultralytics' bundled ``zidane.jpg`` (which has two people in it) and animate
them independently across a canvas. One crop is hue-shifted so the two
"fighters" are visually distinct -- this matters here specifically because
``fighter_identity.py`` tells them apart by appearance histogram, and two
near-identical crops would collapse into one gallery slot.

Offline, deterministic, no network. Not a scoring-quality signal -- it only
proves the multi-person tracking/identity/event pipeline runs end to end.

Usage:
    python scripts/make_two_person_fixture.py
    python scripts/make_two_person_fixture.py --frames 300 --output cache/fixtures/two_person.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _resolve_project_path(path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _person_crop() -> np.ndarray:
    """Crop the single highest-confidence person box out of ultralytics' seed image.

    zidane.jpg has two people, but they stand close enough together that
    their boxes overlap and each crop pulls in the other person too -- their
    color histograms then read as near-identical, which defeats the point of
    this fixture. Using one clean crop and deriving the second "fighter" via
    ``_hue_shift`` (see ``make_two_person_fixture``) sidesteps that entirely.
    """
    import ultralytics
    from ultralytics import YOLO

    seed = Path(ultralytics.__file__).resolve().parent / "assets" / "zidane.jpg"
    img = cv2.imread(str(seed))
    if img is None:
        sys.exit(f"could not read fixture seed image: {seed}")

    model = YOLO(str(PROJECT_ROOT / "yolo26n-pose.pt"))
    res = model.predict(img, imgsz=640, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        sys.exit("pose model found no person in the seed image")

    confs = res.boxes.conf.cpu().numpy()
    boxes = res.boxes.xyxy.cpu().numpy()
    x1, y1, x2, y2 = boxes[int(np.argmax(confs))]
    w, h = x2 - x1, y2 - y1
    mw, mh = 0.12 * w, 0.10 * h
    x1 = max(0, int(x1 - mw))
    y1 = max(0, int(y1 - mh))
    x2 = min(img.shape[1], int(x2 + mw))
    y2 = min(img.shape[0], int(y2 + mh))
    return img[y1:y2, x1:x2]


def _hue_shift(crop: np.ndarray, degrees: int, saturation_boost: float = 2.5) -> np.ndarray:
    """Shift hue and boost saturation so this crop reads as a visually distinct
    "corner color" -- much of a suit-and-skin photo is low-saturation, where hue
    is barely perceptible, so hue-shifting alone isn't enough to separate it from
    the original; real fighters' corner gear is comparatively vivid anyway.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + degrees) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_boost, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def make_two_person_fixture(output: Path, frames: int = 150, width: int = 960, height: int = 540, fps: int = 30) -> int:
    crop_a = _person_crop()
    crop_b = _hue_shift(cv2.flip(crop_a, 1), degrees=90)
    print(f"person crops    a={crop_a.shape[1]}x{crop_a.shape[0]}  b=flipped+hue-shifted duplicate")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        sys.exit(f"could not open VideoWriter for {output}")

    bg = np.full((height, width, 3), 90, dtype=np.uint8)
    bg[: height // 2] = 120  # a horizon line, so the encoder has some structure
    base_h = int(height * 0.72)

    for t in range(frames):
        frame = bg.copy()
        for crop, phase_offset, side in ((crop_a, 0.0, -1), (crop_b, np.pi, 1)):
            phase = 2 * np.pi * t / 90.0 + phase_offset
            scale = 1.0 + 0.08 * np.sin(phase / 2.0)
            h = int(base_h * scale)
            w = max(1, int(crop.shape[1] * h / crop.shape[0]))
            person = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

            # Two independent paths that cross near the middle, to exercise
            # tracking/identity through a close approach (not a full occlusion).
            cx = int(width / 2 + side * 0.22 * width * np.cos(phase / 3.0))
            x = int(np.clip(cx - w // 2, 0, width - 1))
            y = int(np.clip(height - h - 10, 0, height - 1))
            ph, pw = min(h, height - y), min(w, width - x)
            frame[y : y + ph, x : x + pw] = person[:ph, :pw]
        writer.write(frame)

    writer.release()
    cap = cv2.VideoCapture(str(output))
    got = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"fixture         {output} ({width}x{height}, {got} frames @ {fps}fps)")
    return 0 if got > 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=str, default="cache/fixtures/two_person_fixture.mp4")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        make_two_person_fixture(
            _resolve_project_path(args.output), frames=args.frames, width=args.width, height=args.height, fps=args.fps
        )
    )
