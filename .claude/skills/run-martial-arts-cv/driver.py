#!/usr/bin/env python
"""Agent-facing harness for the martial arts pose trainer.

The repo's own docs assume a developer at their own machine: an activated
``.venv``, an ``InputVideo/`` folder of clips, a webcam, and an OpenCV window to
look at. None of that holds for an agent, and none of it is checked into git.
This driver removes every one of those assumptions:

* it finds the project interpreter itself (including from a git worktree, which
  has no ``.venv`` of its own) and re-execs under it;
* it synthesizes a person-bearing test clip offline, so the video pipeline can
  run in a repo that ships no videos;
* it runs the pipeline headless and reports the resulting ``metrics.csv`` as a
  summary you can assert on;
* it renders one overlay frame to a PNG you can actually open;
* it calls the scoring core directly, with no video at all, for the layer most
  changes here actually touch.

Every artifact lands OUTSIDE the repository (see ``out_dir``) so a run never
dirties the working tree. That is not politeness -- see ``keypoints`` in the
notes below.

    python .claude/skills/run-martial-arts-cv/driver.py doctor
    python .claude/skills/run-martial-arts-cv/driver.py fixture
    python .claude/skills/run-martial-arts-cv/driver.py run
    python .claude/skills/run-martial-arts-cv/driver.py shot
    python .claude/skills/run-martial-arts-cv/driver.py score
    python .claude/skills/run-martial-arts-cv/driver.py test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# driver.py lives at <repo>/.claude/skills/run-martial-arts-cv/driver.py
REPO = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("MACV_OUT") or (Path(tempfile.gettempdir()) / "macv-run"))
FIXTURE = OUT / "fixture_person.mp4"
OVERLAY = OUT / "run_overlay.mp4"
SHOT = OUT / "run_frame.png"
STORAGE = OUT / "data"
KPTS = OUT / "keypoints"
RUN_NAME = "driver_run"


# ---------------------------------------------------------------------------
# Interpreter bootstrap
#
# The heavy deps (torch/cv2/ultralytics) live in the project venv, which is
# gitignored -- so a worktree checkout does not have one. Rather than install
# ~3GB of torch per worktree, find the main checkout's venv via
# `git rev-parse --git-common-dir` and re-exec under it.
# ---------------------------------------------------------------------------

def _venv_python(root: Path) -> Path | None:
    for rel in ("Scripts/python.exe", "bin/python"):
        cand = root / ".venv" / rel
        if cand.is_file():
            return cand
    return None


def _main_checkout() -> Path | None:
    """The primary worktree's root, or None when we are already in it."""
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None
    if not common:
        return None
    common_path = Path(common)
    if not common_path.is_absolute():
        common_path = (REPO / common_path).resolve()
    # <main>/.git  ->  <main>
    return common_path.parent if common_path.name == ".git" else None


def find_python() -> Path | None:
    override = os.environ.get("MACV_PYTHON")
    if override:
        p = Path(override)
        return p if p.is_file() else None
    for root in (REPO, _main_checkout()):
        if root is None:
            continue
        found = _venv_python(root)
        if found is not None:
            return found
    return None


def ensure_project_python() -> None:
    """Re-exec under the project venv if the current interpreter lacks the deps."""
    try:
        import cv2  # noqa: F401
        import torch  # noqa: F401
        import ultralytics  # noqa: F401
        return
    except Exception:
        pass

    if os.environ.get("MACV_REEXEC"):
        sys.exit(
            "the project interpreter is missing torch/cv2/ultralytics.\n"
            f"  interpreter: {sys.executable}\n"
            "  fix: point MACV_PYTHON at a python that has them, or\n"
            "       pip install -r requirements.txt into the project .venv"
        )

    override = os.environ.get("MACV_PYTHON")
    if override and not Path(override).is_file():
        # Don't let the generic "no venv found" message below hide the real cause.
        sys.exit(f"MACV_PYTHON is set but is not a file: {override}\n  fix: correct it, or unset it to auto-detect the project venv")

    python = find_python()
    if python is None:
        sys.exit(
            "could not find the project venv.\n"
            f"  looked in: {REPO / '.venv'}\n"
            f"             {(_main_checkout() or Path('<no main checkout>')) / '.venv'}\n"
            "  fix: set MACV_PYTHON to the interpreter that has torch/cv2/ultralytics, e.g.\n"
            "       MACV_PYTHON=D:/PrivateProjects/MartialArtsCommputerVision/.venv/Scripts/python.exe"
        )

    env = dict(os.environ, MACV_REEXEC="1")
    sys.exit(subprocess.run([str(python), str(Path(__file__).resolve()), *sys.argv[1:]], env=env).returncode)


ensure_project_python()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import ultralytics  # noqa: E402


def _import_app():
    """Import action_recognition from the repo (not from wherever cwd is)."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import action_recognition as ar  # noqa: PLC0415
    return ar


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

def cmd_doctor(args: argparse.Namespace) -> int:
    print(f"repo            {REPO}")
    print(f"out dir         {OUT}")
    print(f"interpreter     {sys.executable}")
    print(f"python          {sys.version.split()[0]}")
    print(f"torch           {torch.__version__}")
    print(f"cv2             {cv2.__version__}")
    print(f"numpy           {np.__version__}")
    print(f"ultralytics     {ultralytics.__version__}")
    cuda = torch.cuda.is_available()
    print(f"cuda            {cuda}" + (f" ({torch.cuda.get_device_name(0)})" if cuda else " (CPU only -- runs, just slower)"))

    ok = True
    weights = REPO / "yolo26n-pose.pt"
    if weights.is_file():
        print(f"weights         {weights.name} ({weights.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"weights         MISSING at {weights}")
        ok = False

    assets = Path(ultralytics.__file__).resolve().parent / "assets"
    seed = assets / "zidane.jpg"
    print(f"fixture seed    {'ok' if seed.is_file() else 'MISSING'} {seed}")
    ok = ok and seed.is_file()

    ar = _import_app()
    refs = ar.load_reference_pose_library(str(REPO / "reference_poses"))
    total = sum(len(v) for v in refs.values())
    print(f"reference bank  {len(refs)} techniques / {total} references")
    for tech in sorted(refs):
        print(f"                {tech:18s} {len(refs[tech]):3d}")
    if not refs:
        print("                MISSING -- reference_poses/<technique>/<angle>.npy is empty")
        ok = False

    tracks = sorted((REPO / "keypoints").glob("track_*.npy"))
    print(f"keypoint windows {len(tracks)} committed fixtures in keypoints/")
    ok = ok and bool(tracks)

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# fixture
#
# The repo ships no videos (InputVideo/ and Golden_Seeds/ are gitignored and
# absent), so there is nothing to feed the pipeline. Build a clip instead:
# take the highest-confidence person the pose model finds in ultralytics'
# bundled zidane.jpg, and animate that crop across a canvas. Offline,
# deterministic, and the tracker sees exactly one person.
# ---------------------------------------------------------------------------

def cmd_fixture(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    OUT.mkdir(parents=True, exist_ok=True)
    seed = Path(ultralytics.__file__).resolve().parent / "assets" / "zidane.jpg"
    img = cv2.imread(str(seed))
    if img is None:
        sys.exit(f"could not read fixture seed image: {seed}")

    model = YOLO(str(REPO / "yolo26n-pose.pt"))
    res = model.predict(img, imgsz=640, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        sys.exit("pose model found no person in the seed image")

    confs = res.boxes.conf.cpu().numpy()
    box = res.boxes.xyxy.cpu().numpy()[int(np.argmax(confs))]
    x1, y1, x2, y2 = box
    mw, mh = 0.12 * (x2 - x1), 0.10 * (y2 - y1)
    x1 = max(0, int(x1 - mw)); y1 = max(0, int(y1 - mh))
    x2 = min(img.shape[1], int(x2 + mw)); y2 = min(img.shape[0], int(y2 + mh))
    crop = img[y1:y2, x1:x2]
    print(f"person crop     {crop.shape[1]}x{crop.shape[0]} (conf {confs.max():.2f})")

    W, H, N, FPS = 960, 540, args.frames, 30
    base_h = int(H * 0.82)
    writer = cv2.VideoWriter(str(FIXTURE), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        sys.exit(f"could not open VideoWriter for {FIXTURE}")

    bg = np.zeros((H, W, 3), np.uint8)
    bg[:] = 90
    bg[: H // 2] = 120  # a horizon line, so the encoder has some structure

    for t in range(N):
        frame = bg.copy()
        phase = 2 * np.pi * t
        scale = 1.0 + 0.10 * np.sin(phase / 60.0)
        h = int(base_h * scale)
        w = max(1, int(crop.shape[1] * h / crop.shape[0]))
        person = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

        cx = int(W / 2 + 0.16 * W * np.sin(phase / 90.0))
        x = int(np.clip(cx - w // 2, 0, W - 1))
        y = int(np.clip(H - h - 10 + 14 * np.sin(phase / 45.0), 0, H - 1))
        ph, pw = min(h, H - y), min(w, W - x)
        frame[y : y + ph, x : x + pw] = person[:ph, :pw]
        writer.write(frame)

    writer.release()
    cap = cv2.VideoCapture(str(FIXTURE))
    got = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"fixture         {FIXTURE} ({W}x{H}, {got} frames @ {FPS}fps)")
    return 0 if got > 0 else 1


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def _run_dir() -> Path:
    return STORAGE / "runs" / RUN_NAME


def _quiet(text: str) -> list[str]:
    """Drop ultralytics' redrawing progress bar, keep the real output.

    Belt and braces: the child already runs with ``YOLO_VERBOSE=False``, which
    disables the bar at the source. A differently-configured ultralytics still
    gets filtered here instead of dumping a screenful of escape codes.
    """
    lines = []
    for raw in text.replace("\x1b[K", "\n").replace("\r", "\n").splitlines():
        line = raw.rstrip()
        if not line or line.startswith("processing frames:"):
            continue
        lines.append(line)
    return lines


def _read_metrics() -> list[dict]:
    path = _run_dir() / "metrics.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cmd_run(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve() if args.source else FIXTURE
    if not source.is_file():
        sys.exit(f"no source video at {source}\n  fix: run `driver.py fixture` first, or pass --source <video>")

    # Wipe the previous run so the summary below can't read stale rows: the app
    # APPENDS to metrics.csv when a run name is reused.
    run_dir = _run_dir()
    if run_dir.exists():
        for p in sorted(run_dir.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        run_dir.rmdir()
    OUT.mkdir(parents=True, exist_ok=True)

    # Every path here is ABSOLUTE on purpose: action_recognition resolves
    # relative path flags against the REPO root (_resolve_project_path), not
    # your cwd -- so a relative --save-kpts-dir would overwrite the committed
    # keypoints/track_*.npy fixtures the tests depend on.
    cmd = [
        sys.executable, str(REPO / "action_recognition.py"),
        "--source", str(source),
        "--target-technique", args.technique,
        "--reference-dir", str(REPO / "reference_poses"),
        "--weights", str(REPO / "yolo26n-pose.pt"),
        "--output-path", str(OVERLAY),
        "--save-kpts-dir", str(KPTS),
        "--storage-root", str(STORAGE),
        "--run-name", RUN_NAME,
        "--num-video-sequence-samples", str(args.window),
        "--disable-video-classifier",
        "--no-display",
    ]
    if args.clean:
        # The `fighter id N | activity X` box label is drawn straight over the
        # trainer panel's third line. Drop the boxes when the panel text matters.
        cmd.append("--no-boxes")
    if args.debug:
        cmd.append("--debug")

    log = OUT / "run.log"
    print("$ " + " ".join(cmd), flush=True)
    print(f"(~1 min for 120 frames; full child output -> {log})", flush=True)
    # YOLO_VERBOSE=False turns off ultralytics' progress bar, which otherwise
    # redraws hundreds of times and buries the output that matters.
    env = dict(os.environ, YOLO_VERBOSE="False")
    proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, errors="replace")
    log.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
    for line in _quiet(proc.stdout or ""):
        print(line)
    if proc.returncode != 0:
        print("\n".join(_quiet(proc.stderr or "")[-40:]) or (proc.stderr or "")[-4000:])
        return proc.returncode

    rows = _read_metrics()
    print()
    print(f"technique       {args.technique}")
    print(f"overlay video   {OVERLAY}")
    print(f"run artifacts   {run_dir}")
    if not rows:
        print("scored frames   0  <-- pipeline ran but never scored; see Troubleshooting in SKILL.md")
        return 1

    scores = [float(r["score"]) for r in rows]
    correct = sum(1 for r in rows if r["is_correct"] == "True")
    angles: dict[str, int] = {}
    for r in rows:
        angles[r["reference_angle"]] = angles.get(r["reference_angle"], 0) + 1
    top = sorted(angles.items(), key=lambda kv: -kv[1])[:3]
    print(f"scored frames   {len(rows)} (frames {rows[0]['frame']}..{rows[-1]['frame']})")
    # 2dp on purpose: at 1dp two different techniques on the same fixture print
    # identical min/max and look like a bug that isn't there.
    print(f"score           min {min(scores):.2f} / mean {sum(scores) / len(scores):.2f} / max {max(scores):.2f}"
          f"  (threshold {float(rows[0]['score_threshold']):.0f})")
    print(f"is_correct      {correct}/{len(rows)}")
    print(f"best angles     " + ", ".join(f"{a} x{n}" for a, n in top))
    print(f"last feedback   {rows[-1]['feedback_1'] or '-'} | {rows[-1]['feedback_2'] or '-'}")
    return 0


# ---------------------------------------------------------------------------
# shot
# ---------------------------------------------------------------------------

def cmd_shot(args: argparse.Namespace) -> int:
    if not OVERLAY.is_file():
        sys.exit(f"no overlay video at {OVERLAY}\n  fix: run `driver.py run` first")

    cap = cv2.VideoCapture(str(OVERLAY))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.frame is not None:
        idx = args.frame
        why = "requested"
    else:
        rows = _read_metrics()
        if rows:
            # A scored frame is the interesting one: score panel populated, and
            # the ghost pose drawn if the score is under threshold.
            pick = max(rows, key=lambda r: float(r["score"]))
            idx, why = int(pick["frame"]), f"best-scoring frame (score {float(pick['score']):.1f})"
        else:
            idx, why = total // 2, "midpoint (no metrics)"
    idx = max(0, min(idx, total - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        sys.exit(f"could not read frame {idx} of {OVERLAY}")

    out = Path(args.output).resolve() if args.output else SHOT
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"frame {idx}/{total} ({why}) -> {out}  [{frame.shape[1]}x{frame.shape[0]}]")
    return 0


# ---------------------------------------------------------------------------
# score -- direct invocation, no video, no subprocess
#
# This is the layer most changes here touch (see the scoring-core speedup
# work). It calls _best_reference_match exactly the way run() does, on a
# committed keypoint window.
# ---------------------------------------------------------------------------

def _user_window(ar, samples: int) -> tuple[np.ndarray, Path]:
    """The longest committed track window, trimmed the way run() feeds the scorer."""
    best, best_fp = None, None
    for fp in sorted((REPO / "keypoints").glob("track_*.npy")):
        arr = np.load(fp)
        if arr.ndim == 3 and arr.shape[1] >= 10 and (best is None or arr.shape[0] > best.shape[0]):
            best, best_fp = arr.astype(np.float32), fp
    if best is None:
        sys.exit(f"no usable keypoint window in {REPO / 'keypoints'}")
    return (best[-samples:] if best.shape[0] >= samples else best), best_fp


def cmd_score(args: argparse.Namespace) -> int:
    ar = _import_app()
    refs = ar.load_reference_pose_library(str(REPO / "reference_poses"))
    window, fp = _user_window(ar, args.window)
    print(f"user window     {fp.name} shape {window.shape}")

    techniques = [args.technique] if args.technique else sorted(refs)
    rows = []
    for tech in techniques:
        bank = refs.get(tech)
        if not bank:
            print(f"{tech}: no references")
            continue
        best = ar._best_reference_match(
            user_sequence=window, reference_bank=bank, technique=tech, topk=args.topk
        )
        if best is None:
            print(f"{tech}: no match")
            continue
        angle, m = best
        rows.append((tech, angle, m))
        print(
            f"{tech:18s} refs {len(bank):3d}  best {angle:22s} "
            f"score {float(m['score']):6.2f}  cos {float(m['cosine_similarity']):6.3f}  "
            f"dtw {float(m['dtw_distance']):7.4f}  angle_err {float(m['angle_error']):7.3f}  "
            f"mirror {bool(m['use_mirror'])}"
        )

    if args.json and rows:
        payload = [
            {"technique": t, "angle": a, **{k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in m.items()}}
            for t, a, m in rows
        ]
        print(json.dumps(payload, indent=2))
    return 0 if rows else 1


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------

def cmd_test(args: argparse.Namespace) -> int:
    rc = 0
    for script in ("test_scoring_equivalence.py", "benchmark_scoring.py"):
        print(f"\n=== {script} ===", flush=True)
        proc = subprocess.run([sys.executable, str(REPO / script)], cwd=str(REPO))
        rc = rc or proc.returncode
    return rc


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check interpreter, deps, weights, reference bank").set_defaults(fn=cmd_doctor)

    f = sub.add_parser("fixture", help="synthesize an offline person video to drive the pipeline")
    f.add_argument("--frames", type=int, default=120)
    f.set_defaults(fn=cmd_fixture)

    r = sub.add_parser("run", help="run the full trainer pipeline headless and summarize metrics.csv")
    r.add_argument("--source", default=None, help="video file (default: the synthesized fixture)")
    r.add_argument("--technique", default="jab")
    r.add_argument("--window", type=int, default=8, help="--num-video-sequence-samples")
    r.add_argument("--clean", action="store_true", help="pass --no-boxes so the box label stops covering the trainer panel")
    r.add_argument("--debug", action="store_true")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("shot", help="write one overlay frame to a PNG you can open")
    s.add_argument("--frame", type=int, default=None, help="frame index (default: best-scoring frame)")
    s.add_argument("--output", default=None)
    s.set_defaults(fn=cmd_shot)

    sc = sub.add_parser("score", help="call the scoring core directly on a committed keypoint window")
    sc.add_argument("--technique", default=None, help="default: every technique in the bank")
    sc.add_argument("--window", type=int, default=8)
    sc.add_argument("--topk", type=int, default=0)
    sc.add_argument("--json", action="store_true")
    sc.set_defaults(fn=cmd_score)

    sub.add_parser("test", help="equivalence test + scoring microbenchmark").set_defaults(fn=cmd_test)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
