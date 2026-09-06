# Martial Arts Computer Vision

Reference-based martial arts coaching with pose tracking. A YOLO pose model tracks
the athlete, and each technique is scored frame-by-frame against a library of
captured reference poses, producing a live score, coaching feedback, and a
"ghost" overlay of the target pose.

Bachelor's project. The written report lives in [`report/`](report/).

## Layout

| Path | Contents |
|---|---|
| `action_recognition.py` | Live trainer, reference capture, and evaluation runtime — the main entry point |
| `reference_poses/` | Reference pose library (`<technique>/<angle>.npy`), capture plans, scout output |
| `keypoints/` | Committed pose windows used as fixtures by the benchmark and equivalence test |
| `scripts/` | Data-collection, scouting, batch-run, and analysis tooling |
| `docs/` | Handover guide, runnable command reference, scout architecture notes |
| `report/` | LaTeX thesis source |
| `benchmark_scoring.py` | Video-free microbenchmark for the per-frame scoring core |
| `test_scoring_equivalence.py` | Guards the optimized scoring core against a pinned copy of the original |

## Setup

```bash
python -m venv .venv
```

Activate it (`.\.venv\Scripts\Activate.ps1` on Windows PowerShell), then install
torch for your CUDA version followed by the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

```bash
pip install -r requirements.txt
```

## Running

Score a video against a technique:

```bash
python action_recognition.py --source InputVideo/Jab_Input.mp4 --target-technique jab --disable-video-classifier --no-display
```

Live webcam trainer:

```bash
python action_recognition.py --source 0 --target-technique jab
```

All paths resolve relative to this repository root, so commands work from here
regardless of where the repo is checked out.

See [`docs/ReadyToRunCommands.md`](docs/ReadyToRunCommands.md) for the full
command cookbook and [`docs/HOWTO.md`](docs/HOWTO.md) for the architecture and
handover guide.

## Performance tooling

The per-frame scoring core was the dominant cost — a pure-Python DTW comparing
the live pose window against every reference for the technique. It is now
vectorized and caches reference preprocessing, for roughly a **35x speedup**
with scores unchanged to float rounding.

- `python benchmark_scoring.py` — per-technique ms/call for the scoring core. Needs no video.
- `python test_scoring_equivalence.py` — verifies the optimized path still matches the original. Run this after touching `compare_pose_sequence`, `_best_reference_match`, or `dtw_pose_distance`.
- `python action_recognition.py --profile` — writes a `timing.json` decode/detect/score/draw/encode breakdown plus an optional cProfile `.prof`.

`docs/HOWTO.md` §9.5 documents the opt-in speed flags (`--imgsz`,
`--detect-stride`, `--score-every`, `--score-topk`, `--ref-canonical-len`); all
default to prior behaviour.
