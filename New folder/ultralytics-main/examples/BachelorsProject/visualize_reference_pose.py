from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ultralytics.utils.tqdm import TQDM


COCO17_EDGES = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (5, 6),
    (11, 12),
    (5, 11),
    (6, 12),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a reference pose sequence (.npy) as an animated skeleton."
    )
    parser.add_argument(
        "--reference-file",
        type=str,
        default="",
        help="Path to a saved reference .npy file.",
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="reference_poses",
        help="Root reference directory used with --technique/--angle.",
    )
    parser.add_argument(
        "--technique",
        type=str,
        default="",
        help="Technique subfolder name (example: jab).",
    )
    parser.add_argument(
        "--angle",
        type=str,
        default="",
        help="Angle file name without extension (example: front).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=6.0,
        help="Playback and output FPS.",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.05,
        help="Minimum confidence for drawing joints and edges when confidence exists.",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=720,
        help="Square output canvas size in pixels.",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default="",
        help="Optional output .mp4 path for rendered animation.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Disable live window display. Useful on headless systems.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop playback in the window until you press q.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all .npy references under --reference-dir.",
    )
    parser.add_argument(
        "--batch-output-dir",
        type=str,
        default="reference_poses/previews",
        help="Output directory for batch mp4 previews when --all is used.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate previews even when output files already exist.",
    )
    return parser.parse_args()


def resolve_reference_path(args: argparse.Namespace) -> Path:
    if args.reference_file:
        p = Path(args.reference_file)
        if p.exists():
            return p
        raise FileNotFoundError(f"reference file not found: {p}")

    if not args.technique or not args.angle:
        raise ValueError("provide --reference-file OR both --technique and --angle")

    p = Path(args.reference_dir) / args.technique / f"{args.angle}.npy"
    if not p.exists():
        raise FileNotFoundError(f"reference file not found: {p}")
    return p


def discover_reference_paths(reference_dir: str) -> list[Path]:
    root = Path(reference_dir)
    if not root.exists():
        raise FileNotFoundError(f"reference directory not found: {root}")

    # Only include per-technique angle files, not generated artifacts.
    paths = [
        p
        for p in root.rglob("*.npy")
        if p.parent != root and p.name != "label_thresholds.npy"
    ]
    return sorted(paths)


def validate_sequence(seq: np.ndarray) -> np.ndarray:
    arr = np.asarray(seq, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected (T,K,C), got shape={arr.shape}")
    if arr.shape[2] < 2:
        raise ValueError(f"expected at least 2 channels (x,y), got shape={arr.shape}")
    return arr


def compute_view_transform(seq: np.ndarray, canvas_size: int, margin: int = 40) -> tuple[float, float, float]:
    xy = seq[..., :2]
    finite = np.isfinite(xy).all(axis=-1)
    valid_xy = xy[finite]
    if valid_xy.size == 0:
        return 1.0, canvas_size / 2.0, canvas_size / 2.0

    min_x, min_y = np.min(valid_xy, axis=0)
    max_x, max_y = np.max(valid_xy, axis=0)

    span_x = max(float(max_x - min_x), 1e-6)
    span_y = max(float(max_y - min_y), 1e-6)

    drawable = max(canvas_size - 2 * margin, 1)
    scale = min(drawable / span_x, drawable / span_y)

    tx = margin - float(min_x) * scale
    ty = margin - float(min_y) * scale
    return scale, tx, ty


def draw_frame(frame_kpts: np.ndarray, canvas: np.ndarray, conf_thres: float, scale: float, tx: float, ty: float) -> np.ndarray:
    out = canvas.copy()
    has_conf = frame_kpts.shape[1] >= 3

    def project(i: int) -> tuple[int, int] | None:
        if i >= frame_kpts.shape[0]:
            return None
        x = float(frame_kpts[i, 0])
        y = float(frame_kpts[i, 1])
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        if has_conf:
            c = float(frame_kpts[i, 2])
            if (not np.isfinite(c)) or c < conf_thres:
                return None
        px = int(round(x * scale + tx))
        py = int(round(y * scale + ty))
        return px, py

    for i, j in COCO17_EDGES:
        p1 = project(i)
        p2 = project(j)
        if p1 is None or p2 is None:
            continue
        cv2.line(out, p1, p2, (70, 225, 70), 2, lineType=cv2.LINE_AA)

    for i in range(frame_kpts.shape[0]):
        p = project(i)
        if p is None:
            continue
        cv2.circle(out, p, 4, (0, 220, 255), -1, lineType=cv2.LINE_AA)

    return out


def render_reference(
    ref_path: Path,
    fps: float,
    conf_thres: float,
    canvas_size: int,
    no_window: bool,
    loop: bool,
    save_video: str,
    overwrite: bool,
) -> int:
    seq = validate_sequence(np.load(ref_path))
    t, k, c = seq.shape
    print(f"loaded {ref_path} | shape={seq.shape} | dtype={seq.dtype}")

    scale, tx, ty = compute_view_transform(seq, canvas_size=canvas_size)
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    writer = None
    if save_video:
        out_path = Path(save_video)
        if out_path.exists() and not overwrite:
            print(f"skip existing preview: {out_path}")
            return 2
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (canvas_size, canvas_size))
        if not writer.isOpened():
            raise RuntimeError(f"failed to open output video for writing: {out_path}")

    print(f"frames={t}, keypoints={k}, channels={c}, fps={fps}")
    print("controls: press q to close window")

    delay = max(int(round(1000.0 / max(fps, 0.1))), 1)
    render_progress = TQDM(total=t if not loop else None, desc=f"render {ref_path.parent.name}/{ref_path.stem}", unit="frame")

    while True:
        for idx in range(t):
            frame = draw_frame(seq[idx], canvas, conf_thres, scale, tx, ty)
            label = f"{ref_path.parent.name}/{ref_path.stem}  frame {idx + 1}/{t}"
            cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

            if writer is not None:
                writer.write(frame)

            if not no_window:
                cv2.imshow("Reference Pose Visualizer", frame)
                key = cv2.waitKey(delay) & 0xFF
                if key == ord("q"):
                    if writer is not None:
                        writer.release()
                    render_progress.close()
                    cv2.destroyAllWindows()
                    return 0

            render_progress.update(1)

        if not loop:
            break

    render_progress.close()

    if writer is not None:
        writer.release()
        print(f"saved video: {save_video}")

    if not no_window:
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = parse_args()
    if args.all:
        references = discover_reference_paths(args.reference_dir)
        if not references:
            print(f"no reference .npy files found in {args.reference_dir}")
            return 1

        out_root = Path(args.batch_output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        print(f"batch mode: processing {len(references)} reference file(s)")

        regenerated = 0
        skipped = 0
        failed = 0
        batch_progress = TQDM(total=len(references), desc="preview batch", unit="file")
        for i, ref_path in enumerate(references, start=1):
            technique = ref_path.parent.name
            angle = ref_path.stem
            out_video = out_root / technique / f"{angle}.mp4"
            print(f"[{i}/{len(references)}] {ref_path}")
            if out_video.exists() and not args.overwrite:
                skipped += 1
                print(f"skip existing preview: {out_video}")
                batch_progress.update(1)
                batch_progress.set_postfix(regenerated=regenerated, skipped=skipped, failed=failed)
                continue
            try:
                render_reference(
                    ref_path=ref_path,
                    fps=args.fps,
                    conf_thres=args.conf_thres,
                    canvas_size=args.canvas_size,
                    no_window=args.no_window,
                    loop=False,
                    save_video=str(out_video),
                    overwrite=args.overwrite,
                )
                regenerated += 1
            except Exception as exc:
                failed += 1
                print(f"failed: {ref_path} | {type(exc).__name__}: {exc}")
            batch_progress.update(1)
            batch_progress.set_postfix(regenerated=regenerated, skipped=skipped, failed=failed)

        batch_progress.close()

        print(
            f"batch summary: regenerated={regenerated}, skipped={skipped}, failed={failed}, total={len(references)}"
        )
        print(f"batch previews dir: {out_root}")
        return 0 if failed == 0 else 1

    ref_path = resolve_reference_path(args)
    return render_reference(
        ref_path=ref_path,
        fps=args.fps,
        conf_thres=args.conf_thres,
        canvas_size=args.canvas_size,
        no_window=args.no_window,
        loop=args.loop,
        save_video=args.save_video,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    raise SystemExit(main())
