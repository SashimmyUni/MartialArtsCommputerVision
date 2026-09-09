# Martial Arts Trainer HOWTO

This guide is written for project handover and supervisor presentation.
It explains:

- what the system does end-to-end,
- how data collection and reference capture work,
- how scoring against references is computed,
- how user correction is visualized,
- where the main bottlenecks are and how a supervisor can help.

---

## 1. Project Objective

Build a reference-based martial arts coaching pipeline that can:

1. Track human pose from webcam/video.
2. Compare user motion against canonical reference sequences per technique and camera angle.
3. Output a score and actionable feedback.
4. Show visual correction cues (target ghost pose + correction arrows).

This is a template-matching trainer, not a fully supervised end-to-end classifier.

---

## 2. Folder Structure (What Matters Most)

Root for this project is the repository root itself. All script paths resolve
relative to it, so commands run from there without any `cd` into a subfolder.

Important files and folders:

- `action_recognition.py`
  - Core engine: tracking, pose extraction, scoring, feedback, live overlays, reference recording.
- `scripts/run_reference_collection_batch.py`
  - Batch orchestrator for overnight or large-scale reference capture from the CSV plan.
- `visualize_reference_pose.py`
  - Renders saved `.npy` references into preview videos for quality inspection.
- `generate_reference_capture_commands.py`
  - Builds capture command plans from candidate sources.
- `run_golden_seed_technique.py`
  - Runs all local Golden Seeds files for one technique and saves indexed references automatically.
- `reference_poses/`
  - Canonical reference library and plan files.
- `reference_poses/generated_capture_plan_all_labels.csv`
  - Master capture plan (one row per technique-angle job, with source URLs).
- `reference_poses/<technique>/<angle>.npy`
  - Saved reference motion windows.
- `data/runs/<run_id>/`
  - Structured outputs: config, metrics.csv, tracks, summaries.
- `keypoints/`
  - Raw saved per-track keypoint sequences from runs.

Reference naming model:

- Technique folder: snake_case (example: `front_kick`)
- Angle file: `front`, `left45`, `right45`, `side` (+ optional indexed versions like `right45_03.npy`)

---

## 3. End-to-End Pipeline

### Phase A: Build reference library

- Use `run_reference_collection_batch.py` (preferred for scale) or manual `action_recognition.py --record-reference`.
- Candidate windows are filtered by quality gates.
- Accepted windows are stored as `.npy` reference sequences.

### Phase B: Validate references

- Use `visualize_reference_pose.py` to inspect each saved sequence.
- Remove and recapture references that are static, incomplete, or detection-corrupted.

### Phase C: Run trainer/inference

- Input can be webcam, local video, or YouTube URL.
- For each active track, a pose sequence window is compared to all available angle references of the target technique.
- Best angle is selected automatically.
- Final score + text feedback + visual correction overlay are generated.

---

## 4. How Scripts Work

## 4.1 `run_reference_collection_batch.py`

Purpose:

- Reads `generated_capture_plan_all_labels.csv`.
- Executes all rows with `command_ready=yes`.
- For each technique-angle row, tries to save up to `examples_per_angle` references.

Key behavior:

- Preflight validates number of distinct source URLs.
- Skips rows that already have enough examples (unless `--overwrite`).
- Runs three stages: prefetch sources, extract pose detections once per
  distinct video, then select reference windows from the cache.

It used to launch one `action_recognition.py` subprocess per saved `.npy` —
208 for the ready plan, each re-downloading and re-inferring a video to keep a
single window, and each paying interpreter start, `import torch`, CUDA context
and model load first. Those 208 slots name only 119 distinct videos.

Useful flags:

- `--examples-per-angle` (default 4)
- `--overwrite`
- `--allow-source-reuse`
- `--num-video-sequence-samples`
- `--ref-min-return-closure`
- `--cpu-threads`
- `--preflight-only`
- `--prefetch-only` — download every source, then stop
- `--max-windows-per-video` (default 1) — how many windows one video may
  contribute to a row before other sources are tried
- `--score-topk` (default 0 = exact) — prescreen the reference bank when ranking
- `--legacy` — the original subprocess-per-example flow

## 4.2 `action_recognition.py`

Purpose:

- Main runtime for both:
  - reference capture mode,
  - live trainer scoring mode.

Core internals:

- Loads references with `load_reference_pose_library()`.
- Saves references with `save_reference_pose()`.
- Computes best match via `_best_reference_match()` over all angles in a technique bank.
- Computes score bundle in `compare_pose_sequence()`.
- Generates rule-based textual advice in `generate_feedback()`.
- Generates correction visuals with `_build_reference_overlay()` and `_draw_reference_ghost()`.

## 4.3 `visualize_reference_pose.py`

Purpose:

- QA tool for references.
- Converts `.npy` keypoint sequences into skeleton video previews.
- Helps detect bad captures before they affect scoring.

---

## 5. Data Collection Workflow (Reference Capture)

## 5.0 Golden Seeds folder-first collection (fast local workflow)

Use this when you already have local curated clips under `reference_poses/Golden_Seeds/<TechniqueName>/`.

1. Put clips in one folder per technique (example: `reference_poses/Golden_Seeds/FightingStance/`).
2. Run the per-technique batch script:
  - `python scripts/run_golden_seed_technique.py --technique-key fighting_stance --golden-technique-dir FightingStance --dry-run`
  - `python scripts/run_golden_seed_technique.py --technique-key fighting_stance --golden-technique-dir FightingStance`
3. The script infers angle from each file name and saves outputs to `reference_poses/fighting_stance/` as indexed files:
  - `front_01.npy`, `front_02.npy`, `left45_01.npy`, `behind_01.npy`, etc.
4. Inspect results with `visualize_reference_pose.py` and re-run with `--overwrite` if needed.

## 5.1 Plan-driven collection

1. Maintain source URLs per row in `generated_capture_plan_all_labels.csv`.
2. Launch batch runner.
3. Batch captures references and writes `.npy` files by technique/angle.
4. Review with preview script.
5. Recapture weak references with stricter gates — see 5.3, this no longer
   means re-downloading anything.

## 5.3 The two caches, and why gate re-tuning is now cheap

```
cache/videos/<video_id>.<ext>          prefetched source video
cache/tracks/<video_id>__<sig>.npz     every pose detection the model made
```

Both are gitignored and rebuild from the plan, so `cache/` can be deleted at any
time. The videos dominate the disk cost (tens of GB for a full plan); the
detection tables are small.

`<sig>` covers everything that changes the *detections* — weights, image size,
precision, tracker, ultralytics version, device — and deliberately nothing that
only changes *which window is selected*. That asymmetry is the point: the
`ref_*` gates, `num_video_sequence_samples` and `reference_sequence_mode` are
not in the key, so changing one replays the cached tracks instead of re-running
the model.

```bash
python scripts/select_reference_windows.py --ref-min-return-closure 0.30 --overwrite
```

No GPU, no video decode, no network. To be concrete rather than vague about it:
re-selecting the four jab rows (13 distinct videos) takes about **170 s** of CPU
with exact ranking, or **68 s** with `--score-topk 3`. Nearly all of that is the
DTW sweep of the reference bank — 21 references for jab — while ranking
candidates. So it is minutes for a technique rather than seconds, but it is
minutes without touching the network or the GPU, against hours of re-downloading
and re-inference before. That is what makes 9.1 iterable.

Two consequences worth knowing:

- One video pass can now yield several windows. By default each of a row's
  distinct sources still contributes one example, preserving source diversity;
  `--max-windows-per-video` raises the cap for rows short on sources. A
  similarity ceiling (`--max-self-similarity`, default 95) rejects a window too
  much like one already accepted, so several windows from one video are not the
  same action repeated.
- `test_capture_equivalence.py` pins the replay against a copy of the live
  capture loop. Run it after touching `reference_selection.py`, the window
  extractors, or the gates.

## 5.4 Screening candidates before capture

`scripts/rank_candidates_by_pose.py` scores a candidate video's best available
window against a technique's reference bank with `_best_reference_match` — the
matcher the trainer itself uses, so it accounts for mirroring and joint angles.
This finally wires up the pose-based filtering 9.2 wanted, though not for free:
a candidate must be downloaded and run through the pose model to be scored. It
pays off when that is cheaper than a capture run plus reviewing the weak
references that follow, and the extraction is cached, so a candidate that
survives screening is not re-extracted when it is captured.

Note it does *not* use `scout_utils.compute_pose_match_score`, which was written
for this and left unused: its sequence-length term rewards windows that merely
happen to be the same length — meaningless for variable-length `stance_cycle`
output — and it ignores mirroring and joint angles.

## 5.2 Candidate acceptance gates

During capture, each candidate window must pass:

- Motion gate: `ref_min_motion_energy`
  - Rejects near-static windows.
- Return closure gate: `ref_min_return_closure`
  - Enforces extension + retraction patterns.
- Score gate: `ref_min_score_gate`
  - After bootstrap, candidate should still resemble existing references for that technique.
- Optional Golden Seed variation band: `capture_seed_min_score` + `capture_seed_max_score`
  - Candidate must stay close enough to the seed bank to remain the same technique,
    but below the upper bound so it is not accepted as a near-duplicate of the Golden Seed.

Capture mode options:

- `first_valid`: save first window that passes gates.
- `best_window`: scan and choose best valid window (preferred quality).

---

## 6. How Reference vs Actual Pose Is Calculated

Input:

- User sequence window and one reference sequence.

Normalization:

- Poses are normalized per frame to body-centered coordinates for scale/translation robustness.

Alignment and orientation:

- Sequences are resampled to same length.
- Both plain and mirrored user sequence are compared.
- The better orientation is selected (`use_mirror`).

Metrics in `compare_pose_sequence()`:

- Cosine pose similarity
- DTW pose distance (temporal alignment)
- Technique-specific joint angle error
- Mean pose distance

Final score (0 to 100):

`final_score = 0.35*cosine_score + 0.25*dtw_score + 0.25*angle_score + 0.15*pose_dist_score`

Where each sub-score is converted into a 0 to 100 scale before fusion.

Best-angle selection:

- `_best_reference_match()` evaluates all references of a technique and chooses max score.

Correct/incorrect decision:

- Compare score to threshold (`trainer_score_threshold`, default 70, optionally per-technique overrides).

---

## 7. How Correction Is Indicated to the User

Current correction UX has three layers:

1. Compact info panel
- Technique and matched reference angle
- Current score vs threshold
- Short text feedback

2. Ghost target pose (when incorrect)
- A reference frame is projected into current user image coordinates.
- Drawn as semi-transparent skeleton overlay.

3. Correction arrows
- Focus joints (arms for punches, legs for kicks) are ranked by largest normalized error.
- Top errors are shown with arrows from current joint position to target joint position.

Interpretation:

- Arrow direction = where that joint should move.
- Ghost skeleton = desired pose shape in the current frame context.

---

## 8. Recommended Demo Flow for Supervisor Meeting

Use this order to clearly communicate project value:

1. Show folder structure and artifacts
- `reference_poses/` and `data/runs/` organization.

2. Show one reference preview
- Demonstrates what is considered canonical motion.

3. Run one trainer video test
- Show score, matched angle, and correction overlay behavior.

4. Open `metrics.csv` from that run
- Show quantitative scoring trace over time.

5. Explain bottlenecks and next research/engineering steps
- Section 9 below.

---

## 9. Current Bottlenecks (Supervisor Help Needed)

This is the key discussion section.

## 9.1 Reference quality and timing (highest priority)

Observed issue:

- Some captures still save windows that are too static, mistimed, or not the true peak action moment.

Why this matters:

- Reference quality directly limits trainer reliability.
- A weak reference can bias best-angle matching and feedback quality.

Supervisor support requested:

- Define stricter acceptance criteria per technique (especially kicks).
- Help design an annotation protocol for "true contact/extension" frame windows.
- Approve a small manually curated gold-standard subset for calibration.

## 9.2 Data diversity and coverage

Observed issue:

- Limited performer diversity, camera setups, and motion styles.

Why this matters:

- Reduced robustness across users and recording conditions.

Supervisor support requested:

- Access to more varied source material and/or controlled recording sessions.
- Guidance on minimum dataset size per technique-angle pair.

## 9.3 Threshold calibration and evaluation methodology

Observed issue:

- Single global threshold can be suboptimal; techniques differ in score behavior.

Why this matters:

- False negatives/positives vary by technique.

Supervisor support requested:

- Define evaluation protocol (validation set, metrics, acceptance targets).
- Support per-technique threshold calibration and periodic re-baselining.

## 9.4 Runtime stability of online video sources — addressed

Observed issue:

- YouTube stream interruptions can terminate long runs early.

Why this matters:

- Affects overnight automation reliability.

Supervisor support requested:

- Endorse policy to pre-download videos for batch runs.
- Optionally support local dataset mirroring to avoid stream-side failures.

Update: sources are now downloaded once into `cache/videos/` before capture
starts (`scripts/prefetch_sources.py`, or
`run_reference_collection_batch.py --prefetch-only`), rather than streamed
straight into `cv2.VideoCapture` on every use. Downloading is retryable on its
own instead of failing in the middle of a capture run, and the 60 plan URLs used
by more than one row are fetched once rather than per row. A missing or broken
`yt-dlp` falls back to streaming, so the cache makes capture more reliable
without becoming a hard dependency.

One caveat for the supervisor to sign off on, since 9.4 asks for the policy
rather than the mechanism: cached videos are local, gitignored research
artifacts kept to make capture runs reproducible and to avoid repeated
streaming. Downloading YouTube content may be restricted by its Terms of
Service. Confirm this is acceptable under institutional policy before enabling
the cache, prefer material the project has rights to (the local Golden Seeds
clips) where possible, and do not redistribute cached media. Only the derived
keypoint arrays under `reference_poses/` are committed.

## 9.5 Compute throughput

Observed issue:

- CPU-only runs are slow for large-scale capture/experimentation.

Why this matters:

- Limits iteration speed for data and model tuning.

Supervisor support requested:

- Access to a GPU workstation/server for batch capture and evaluation.

Update: the trainer scoring core (`_best_reference_match` /
`compare_pose_sequence` in `action_recognition.py`) was rewritten to cache
normalized/resampled references at load time and to vectorize DTW/angle-error
scoring — previously the dominant per-frame cost (a Python-level nested loop
re-normalizing every reference and computing DTW cell-by-cell, on every scored
frame). Correctness is verified by `test_scoring_equivalence.py`, which checks
the optimized scoring against a pinned copy of the original implementation and
asserts identical results within float tolerance; run it after touching
anything under the scoring core. Throughput can be measured directly with
`benchmark_scoring.py` (no video needed — uses the committed
`keypoints/track_*.npy` files) and, for a full run, with the new `--profile`
flag (writes `timing.json` with a decode/yolo/scoring/drawing/encode
breakdown, plus an optional cProfile `.prof`).

New opt-in speed flags on `action_recognition.py` (all default to prior
behaviour — pass none of these and nothing changes):

- `--imgsz N` — YOLO inference size (default 640).
- `--detect-stride N` — run YOLO every N frames, reuse the last detection in between.
- `--score-every N` — run trainer scoring every N eligible ticks; the overlay keeps the last score in between.
- `--score-topk K` — cosine-prescreen the reference bank and only DTW-score the top K candidates (0 = score the whole bank).
- `--ref-canonical-len L` — resample every loaded reference to a fixed frame count at load time, shrinking the O(T²) DTW cost per reference (0 = keep native length).
- `--fast-mode` now actually matches its own help text (previously it left
  `visualize_pose`/`draw_boxes`/`overlay_pose` on and never touched
  `skip_frame`, contrary to what it claimed).

`run_reference_collection_batch.py` and `run_inputvideo_batch.ps1` gained a
`--jobs`/`-Jobs` flag to capture multiple rows/videos concurrently instead of
one `action_recognition.py` subprocess at a time (each holds its own YOLO
model + CUDA context, so keep this modest relative to available VRAM).
`run_inputvideo_batch.ps1` also had a path bug fixed: it was `Set-Location`-ing
into `scripts/` and then invoking `action_recognition.py` with a
path relative to that directory, where the script doesn't live — the batch
never actually ran the trainer.

---

## 10. Practical Commands

Run overnight batch:

```powershell
python scripts/run_reference_collection_batch.py
```

Run batch capture using Golden Seed-derived references as a variation band:

```powershell
python scripts/run_reference_collection_batch.py `
  --capture-seed-reference-dir reference_poses `
  --capture-seed-min-score 72 `
  --capture-seed-max-score 94
```

Use this after extracting your curated Golden Seeds into `.npy` references. The lower bound keeps new captures aligned with the seed technique, and the upper bound rejects clips that are effectively the same execution again.

Dry-run preflight only:

```powershell
python scripts/run_reference_collection_batch.py --preflight-only
```

Manual single reference capture:

```powershell
python action_recognition.py `
  --source "https://www.youtube.com/watch?v=VIDEO_ID" `
  --record-reference "front_kick__right45" `
  --target-technique front_kick `
  --reference-capture-mode best_window `
  --num-video-sequence-samples 20 `
  --disable-video-classifier --no-display `
  --auto-exit-after-reference `
  --reference-search-max-frames 1800 `
  --ref-min-motion-energy 0.02 `
  --ref-min-return-closure 0.20 `
  --ref-min-score-gate 0
```

Run trainer on webcam:

```powershell
python action_recognition.py --source 0 --target-technique jab --reference-dir reference_poses
```

Run trainer on a local video:

```powershell
python action_recognition.py --source MultipleTest.MOV --target-technique jab --reference-dir reference_poses --output-path output_demo.mp4 --fast-mode --skip-frame 2
```

Preview one reference:

```powershell
python scripts/visualize_reference_pose.py --technique front_kick --angle right45
```

---

## 11. Suggested Near-Term Plan

1. Freeze and review current references by preview quality.
2. Recapture weakest technique-angle pairs with stricter gates.
3. Build a small validated benchmark set (correct + common mistakes).
4. Calibrate per-technique thresholds from benchmark metrics.
5. Re-run meeting demo with benchmark-backed numbers.

---

If you present only one message to your supervisor:

- The pipeline is already functional and demonstrable.
- Main risk is reference quality at capture time.
- Biggest impact support is in curated data protocol, evaluation methodology, and compute/resources for faster iteration.
