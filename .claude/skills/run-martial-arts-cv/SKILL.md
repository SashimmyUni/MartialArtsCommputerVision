---
name: run-martial-arts-cv
description: Run, launch, drive, or screenshot the martial arts pose trainer (action_recognition.py), and score or benchmark its scoring core. Use when asked to run or start the app, put a video through the YOLO pose pipeline, capture a screenshot of the trainer overlay and ghost pose, verify a scoring/DTW/reference-matching change, or run the equivalence test and microbenchmark.
---

# Running the martial arts pose trainer

`action_recognition.py` tracks an athlete with a YOLO pose model and scores each
frame against a bank of reference poses, drawing a score panel, a coaching tip,
and a "ghost" of the target pose.

Everything here goes through one harness:

    .claude/skills/run-martial-arts-cv/driver.py

It exists because the repo's own docs assume things an agent does not have: an
activated `.venv`, an `InputVideo/` folder of clips, a webcam, and an OpenCV
window to look at. **The repo ships no videos** (`InputVideo/`,
`reference_poses/Golden_Seeds/` are gitignored and absent), so every example
command in `docs/ReadyToRunCommands.md` §2–§6 fails as written. The driver
synthesizes its own input, runs headless, and writes a PNG you can open.

All paths below are relative to the repo root; run from there.

| Command | Time | What you get |
|---|---|---|
| `driver.py doctor` | 3s | interpreter, deps, CUDA, weights, reference-bank inventory |
| `driver.py fixture` | 5s | a synthetic person video (no network, no assets needed) |
| `driver.py run` | 15s warm / 65s cold | full pipeline + a summary of `metrics.csv` |
| `driver.py shot` | 1s | one overlay frame as a PNG |
| `driver.py score` | 4s | the scoring core called directly, no video |
| `driver.py test` | **3m45s** | equivalence test + microbenchmark |

## Prerequisites

The heavy deps live in the project venv at
`D:\PrivateProjects\MartialArtsCommputerVision\.venv` (Python 3.12.10, torch
2.13.0+cu130, ultralytics 8.4.21, cv2 5.0.0, numpy 2.5.2). CUDA works here on a
GTX 1660 SUPER; CPU-only also runs, just slower.

**You do not need to activate it.** `driver.py` finds it and re-execs itself
under it — including from a git worktree, which has no `.venv` of its own (it
resolves the main checkout via `git rev-parse --git-common-dir`). Override with
`MACV_PYTHON=<path-to-python>` if you need a different interpreter.

Creating a venv from scratch is documented in `README.md` (torch from the CUDA
index first, then `requirements.txt`). That path was **not** exercised here —
the venv already existed.

Verify before anything else:

```bash
python .claude/skills/run-martial-arts-cv/driver.py doctor
```

Exits 0 and ends in `OK`. It should report 10 techniques / 93 references and 56
committed keypoint windows.

## Run: the video pipeline (agent path)

```bash
python .claude/skills/run-martial-arts-cv/driver.py fixture
```

Crops the highest-confidence person out of ultralytics' bundled `zidane.jpg` and
animates it across a 960x540 canvas for 120 frames. Offline, deterministic, and
the tracker sees exactly one person.

```bash
python .claude/skills/run-martial-arts-cv/driver.py run --clean
```

Runs `action_recognition.py` end to end and prints:

```
technique       jab
scored frames   113 (frames 8..120)
score           min 48.23 / mean 50.83 / max 53.63  (threshold 70)
is_correct      0/113
best angles     jab3_01 x41, jab2_01 x36, jab_r1_01 x31
last feedback   Extend your punching arm more | Keep your guard hand higher
```

That summary is read back out of `metrics.csv` — assert on it, but on the shape
and the band, not on exact digits: GPU float nondeterminism moves the last
decimal between runs (`mean 50.83` / `50.84` for the same input). Useful flags:
`--technique <name>` (default `jab`), `--source <video>` to use a real clip
instead of the fixture, `--window N` for `--num-video-sequence-samples`,
`--debug`, and `--clean` (see Gotchas — without it a box label covers the
trainer panel).

Artifacts land **outside** the repo, under `%TEMP%\macv-run` (override with
`MACV_OUT`): `run_overlay.mp4`, `run.log` (full child output),
`data/runs/driver_run/{metrics.csv,config.json,timing.json,tracks/}`.

```bash
python .claude/skills/run-martial-arts-cv/driver.py shot
```

Writes `%TEMP%\macv-run\run_frame.png` — by default the best-scoring frame, so
the panel is populated. **Open it.** You should see the tracked person, an
orange pose skeleton, a green ghost skeleton, and a panel reading
`id 1 | jab | ref jab2_01 / score 51.4/100 (target 70.0) / tip: ...`. Add
`--frame N` for a specific frame or `--output <path>` to put it elsewhere.

## Run: the scoring core directly (no video)

This is the layer most changes here touch — `compare_pose_sequence`,
`_best_reference_match`, `dtw_pose_distance`. It needs no video and no
subprocess: it loads a committed `keypoints/track_*.npy` window and the
reference bank, and calls the scorer the same way `run()` does.

```bash
python .claude/skills/run-martial-arts-cv/driver.py score
```

```
user window     track_1.npy shape (8, 17, 3)
axe_kick           refs   8  best axekick2_01   score  64.79  cos  0.920  dtw  0.5869  angle_err  16.068  mirror False
fighting_stance    refs   9  best left45_02     score  73.86  cos  0.977  dtw  0.3089  angle_err  37.054  mirror False
jab                refs  21  best jab2_01       score  65.31  cos  0.932  dtw  0.4115  angle_err  46.534  mirror False
knee_kick          refs   7  best knee2_01      score  85.26  cos  0.979  dtw  0.2620  angle_err   4.558  mirror False
```

`--technique jab` for one bank, `--topk N` to exercise the cosine prescreen,
`--json` for machine-readable output.

## Test

```bash
python .claude/skills/run-martial-arts-cv/driver.py test
```

Runs `test_scoring_equivalence.py` (744 pairwise + 80 best-match checks against
a pinned copy of the pre-optimization scorer) then `benchmark_scoring.py`
(per-technique ms/call). **Takes 3m45s** — it is not hung. Run it after touching
the scoring core.

## Run: human path

`python action_recognition.py --source 0 --target-technique jab` opens OpenCV
windows and needs a webcam and a display. Not usable from here, and **not
exercised in this session** — use the driver instead.

## Gotchas

1. **Never invoke `action_recognition.py` bare.** `save_kpts_dir` defaults to
   `PROJECT_ROOT/"keypoints"` (`action_recognition.py:2432`), and
   `keypoints/track_*.npy` are the committed fixtures that
   `benchmark_scoring.py` and `test_scoring_equivalence.py` read. A plain run
   silently overwrites them. The driver redirects `--save-kpts-dir` every time.
2. **Relative path flags resolve against the repo, not your cwd.**
   `_resolve_project_path` (`action_recognition.py:46`) is applied to
   `--weights`, `--reference-dir`, `--storage-root`, `--output-path`,
   `--save-kpts-dir` and `--source` at lines 2339–2359. `--storage-root data`
   writes into the repo no matter where you invoke it from. The driver passes
   absolute paths for all of them.
3. **The fixture is a still photo being panned**, so the pose never changes.
   Scores sit in a narrow ~48–54 band for *every* technique and never cross the
   70 threshold. It proves the pipeline runs end to end; it is **not** a
   scoring-quality signal. For that use `driver.py score`, which discriminates
   properly (48–85 across banks) because it reads real captured footage.
4. **The first run is ~4x slower** — 63s vs 16s for the same 120 frames (warmup
   dominates: `yolo` reports 420ms/call cold, 27ms/call warm). Don't read the
   first `timing.json` as steady state.
5. **`YOLO_VERBOSE=False` is required** or ultralytics' progress bar redraws
   hundreds of times and glues itself onto real log lines
   (`...120/120 1:03saving keypoints: ...`). The driver sets it and filters
   whatever survives.
6. **The `fighter id N | activity X` box label is drawn over the trainer
   panel's third line**, hiding the coaching tip. Use `run --clean`
   (`--no-boxes`) whenever the panel text matters.
7. **The ghost pose only renders when the score is below threshold** — it is
   gated on `not is_correct` (`action_recognition.py:3016`). A screenshot with
   no green skeleton means the athlete scored as correct, not that the overlay
   broke.
8. **`--disable-video-classifier` is effectively mandatory.** Without it the
   X-CLIP classifier initializes and pulls weights from HuggingFace. The driver
   always passes it.
9. **`metrics.csv` is appended to** when a run name is reused, so a second run
   would double the rows. The driver wipes `data/runs/driver_run/` first.
10. Not covered here: the YouTube scout (`scripts/scout_youtube_by_golden_seeds.py`),
    `merge_scout_into_plan.py`, and `run_reference_collection_batch.py`. They
    need a `YOUTUBE_API_KEY` plus network, and batch collection also needs
    `Golden_Seeds/` videos that are not on disk. Not exercised.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `MACV_PYTHON is set but is not a file: ...` | Correct it, or unset it and let the driver auto-detect. |
| `could not find the project venv` | Neither `<repo>/.venv` nor the main checkout's exists. Set `MACV_PYTHON` to an interpreter that has torch/cv2/ultralytics. |
| `the project interpreter is missing torch/cv2/ultralytics` | The interpreter it found (or you forced) is the wrong one — e.g. system Python 3.13 here has only numpy and pillow. |
| `no source video at ...` | Run `driver.py fixture` first, or pass `--source <video>`. |
| `scored frames 0` after a clean exit | The pipeline ran but never scored: no person detected, or the clip is shorter than `--window` (needs `num_video_sequence_samples` tracked frames before the first score — with the default 8, scoring starts at frame 8). Check `run.log`. |
| `driver.py test` looks hung | It isn't; it takes 3m45s. |
