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

Root for this project:

`ultralytics-main/examples/BachelorsProject/`

Important files and folders:

- `action_recognition.py`
  - Core engine: tracking, pose extraction, scoring, feedback, live overlays, reference recording.
- `run_reference_collection_batch.py`
  - Batch orchestrator for overnight or large-scale reference capture from the CSV plan.
- `visualize_reference_pose.py`
  - Renders saved `.npy` references into preview videos for quality inspection.
- `generate_reference_capture_commands.py`
  - Builds capture command plans from candidate sources.
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
- Runs child capture calls to `action_recognition.py` in reference-capture mode.
- Applies cooldown between jobs to reduce sustained load.

Useful flags:

- `--examples-per-angle` (default 4)
- `--overwrite`
- `--allow-source-reuse`
- `--num-video-sequence-samples`
- `--ref-min-return-closure`
- `--cpu-threads`
- `--preflight-only`

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

## 5.1 Plan-driven collection

1. Maintain source URLs per row in `generated_capture_plan_all_labels.csv`.
2. Launch batch runner.
3. Batch captures references and writes `.npy` files by technique/angle.
4. Review with preview script.
5. Recapture weak references with stricter gates.

## 5.2 Candidate acceptance gates

During capture, each candidate window must pass:

- Motion gate: `ref_min_motion_energy`
  - Rejects near-static windows.
- Return closure gate: `ref_min_return_closure`
  - Enforces extension + retraction patterns.
- Score gate: `ref_min_score_gate`
  - After bootstrap, candidate should still resemble existing references for that technique.

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

## 9.4 Runtime stability of online video sources

Observed issue:

- YouTube stream interruptions can terminate long runs early.

Why this matters:

- Affects overnight automation reliability.

Supervisor support requested:

- Endorse policy to pre-download videos for batch runs.
- Optionally support local dataset mirroring to avoid stream-side failures.

## 9.5 Compute throughput

Observed issue:

- CPU-only runs are slow for large-scale capture/experimentation.

Why this matters:

- Limits iteration speed for data and model tuning.

Supervisor support requested:

- Access to a GPU workstation/server for batch capture and evaluation.

---

## 10. Practical Commands

Run overnight batch:

```powershell
python run_reference_collection_batch.py
```

Dry-run preflight only:

```powershell
python run_reference_collection_batch.py --preflight-only
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
python visualize_reference_pose.py --technique front_kick --angle right45
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
