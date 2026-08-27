# MartialArtsCommputerVision

Workspace for the bachelor's project on reference-based martial arts coaching with pose tracking.

## Active project

The active implementation is under:

- `New folder/ultralytics-main/examples/BachelorsProject/`

## Main entry points

- `New folder/ultralytics-main/examples/BachelorsProject/action_recognition.py`
- `New folder/ultralytics-main/examples/BachelorsProject/run_reference_collection_batch.py`
- `New folder/ultralytics-main/examples/BachelorsProject/run_golden_seed_technique.py`
- `New folder/ultralytics-main/examples/BachelorsProject/ReadyToRunCommands.md`

## Environment

- Python virtual environment: `.venv/` (Python 3.12, CUDA-enabled torch, vendored `ultralytics` installed editable from `New folder/ultralytics-main/`)
- Active reference library: `New folder/ultralytics-main/examples/BachelorsProject/reference_poses/`

## Performance tooling

- `benchmark_scoring.py` — video-free microbenchmark for the trainer's per-frame scoring core (uses the committed `keypoints/track_*.npy` files).
- `test_scoring_equivalence.py` — checks the optimized scoring core against a pinned copy of the original implementation; run this after touching anything under `compare_pose_sequence`/`_best_reference_match`/`dtw_pose_distance`.
- `action_recognition.py --profile` — writes a `timing.json` decode/yolo/scoring/drawing/encode breakdown (and an optional cProfile `.prof`) for a real run.

See `New folder/ultralytics-main/examples/BachelorsProject/Guides/HOWTO.md` §9.5 for the full list of opt-in speed flags (`--imgsz`, `--detect-stride`, `--score-every`, `--score-topk`, `--ref-canonical-len`) — all default to prior behaviour.

