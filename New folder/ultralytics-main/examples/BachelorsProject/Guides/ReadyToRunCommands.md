# Ready To Run Commands

Use these commands from this folder:

- `ultralytics-main/examples/BachelorsProject`

## 1) Optional environment setup (Windows PowerShell)

```powershell
cd "c:\Users\Sashi\Documents\Bachelor\MartialArtsComputerVision\MartialArtsCommputerVision"

# If your venv is at repo root:
& ".\.venv\Scripts\Activate.ps1"

cd ".\New folder\ultralytics-main\examples\BachelorsProject"
```

## 2) Fast trainer run on local video (recommended)

```powershell
python action_recognition.py --source "InputVideo\Jab.mp4" --target-technique jab --reference-dir reference_poses
```

## 3) Trainer run with pose visualization video output

```powershell
python action_recognition.py --source "InputVideo\Jab_Pro_Done.mp4" --target-technique jab --reference-dir reference_poses --output-path Jab_Pro.mp4 --disable-video-classifier --visualize-pose --pose-output-path datasets/pose_jab_pro.mp4 --no-display
```

## 4) Webcam trainer

```powershell
python action_recognition.py --source 0 --target-technique jab --reference-dir reference_poses
```

## 5) Manual reference capture (best window)

```powershell
python action_recognition.py `
  --source "https://www.youtube.com/watch?v=VIDEO_ID" `
  --record-reference "jab__front" `
  --target-technique jab `
  --reference-dir reference_poses `
  --reference-capture-mode best_window `
  --num-video-sequence-samples 150 `
  --reference-capture-buffer-multiplier 3 `
  --disable-video-classifier --no-display `
  --auto-exit-after-reference `
  --reference-search-max-frames 1800 `
  --ref-min-motion-energy 0.1 `
  --ref-min-return-closure 0.0 `
  --ref-min-score-gate 0
```

## 6) Dynamic stance-cycle capture (leave stance -> return stance)

```powershell
python action_recognition.py `
   --source "reference_poses/Golden_Seeds/Jab/FrontJab_Right.MOV" `
  --record-reference "jab__frontjab_right_stancecycle" `
  --target-technique _capture_only `
  --reference-dir reference_poses `
  --reference-capture-mode best_window `
  --reference-sequence-mode stance_cycle `
  --num-video-sequence-samples 150 `
  --reference-capture-buffer-multiplier 3 `
  --disable-video-classifier --no-display `
  --auto-exit-after-reference `
  --reference-search-max-frames 400 `
  --ref-min-motion-energy 0.08 `
  --ref-min-return-closure 0.0 `
  --ref-min-score-gate 0 `
  --ref-stance-start-threshold 0.18 `
  --ref-stance-end-threshold 0.12 `
  --ref-stance-peak-threshold 0.30 `
  --ref-stance-min-frames 24 `
  --ref-stance-hold-frames 4
```

## 7) Preview one reference pose

```powershell
python visualize_reference_pose.py --reference-file reference_poses/front_kick/front_01.npy --save-video reference_poses/previews/front_kick/front_01.mp4 --no-window --overwrite
```

## 8) Preview all references under a folder

```powershell
python visualize_reference_pose.py --all --reference-dir reference_poses --batch-output-dir reference_poses/previews --no-window --overwrite
```

## 9) Batch collection from CSV plan

```powershell
python run_reference_collection_batch.py
```

Dry run only:

```powershell
python run_reference_collection_batch.py --preflight-only
```

## 10) Run all Golden Seeds files for one technique (auto-indexed)

Dry run first:

```powershell
python run_golden_seed_technique.py --technique-key jab --golden-technique-dir Jab --dry-run
```

Run capture:

```powershell
python run_golden_seed_technique.py --technique-key fighting_stance --golden-technique-dir FightingStance
```

Example for AxeKick:

```powershell
python run_golden_seed_technique.py --technique-key axe_kick --golden-technique-dir AxeKick
```

## 11) Jab seed-gated batch helper script

```powershell
powershell -ExecutionPolicy Bypass -File .\run_jab_seed_batch.ps1 -ExamplesPerAngle 2 -NumVideoSequenceSamples 150 -RefMinScoreGate 55 -ReferenceSearchMaxFrames 260
```

## 12) GOLDEN SEEDS YOUTUBE SCOUT (NEW!)

Scout YouTube videos using Golden Seeds as templates. The scout uses the camera angles and variations in your Golden_Seeds folder to automatically search YouTube, find similar technique videos, and populate the batch collection plan.

### Quick Start

Set your YouTube API key:
```powershell
$env:YOUTUBE_API_KEY = "<your-api-key>"
```

Scout all Golden Seeds techniques/angles:
```powershell
python scout_youtube_by_golden_seeds.py --api-key $env:YOUTUBE_API_KEY
```

Scout just one technique/angle:
```powershell
python scout_youtube_by_golden_seeds.py --api-key $env:YOUTUBE_API_KEY --technique jab --angle side_right
```

Dry run (shows queries without API calls):
```powershell
python scout_youtube_by_golden_seeds.py --api-key $env:YOUTUBE_API_KEY --dry-run
```

### Scout Options

```powershell
python scout_youtube_by_golden_seeds.py --api-key $env:YOUTUBE_API_KEY `
  --technique fighting_stance `
  --max-results-per-query 50 `
  --max-pages-per-query 2 `
  --max-candidates-per-angle 4 `
  --min-view-count 1000 `
  --order relevance
```

### Scout Workflow

1. **Scout YouTube**: 
   ```powershell
   python scout_youtube_by_golden_seeds.py --api-key $env:YOUTUBE_API_KEY
   ```
   - Outputs: `reference_poses/scout_candidates_golden_seeds.csv` (detailed results for review)
   - Outputs: `reference_poses/scout_batch_plan.csv` (ready for batch collection)

2. **Review candidates** (optional):
   - Open `reference_poses/scout_candidates_golden_seeds.csv`
   - Check view counts, definitions, captions
   - Manually remove low-quality videos if needed

3. **Merge into batch plan**:
   ```powershell
   python merge_scout_into_plan.py --scout-plan reference_poses/scout_batch_plan.csv --backup
   ```
   - `--backup`: Creates backup of existing plan
   - `--merge-mode update`: Update existing entries, keep user edits for other entries
   - `--merge-mode append`: Add scout results to existing plan
   - `--merge-mode replace`: Completely replace plan with scout results

4. **Run batch collection** (same as #9):
   ```powershell
   python run_reference_collection_batch.py
   ```

### Scout Architecture

The scout system works like this:

1. **Scan Golden Seeds**: Finds all videos in `reference_poses/Golden_Seeds/`
2. **Infer camera angles**: Filename analysis determines angle (front, side_right, left45, etc.)
3. **Generate search queries**: For each technique/angle, generates multiple YouTube search queries
4. **Search YouTube**: Uses YouTube Data API v3 to find matching videos
5. **Filter candidates**: 
   - Duration: 10-600 seconds (adjustable)
   - Views: Minimum 1000 (adjustable)
   - Quality: Prefers HD and videos with captions
6. **Rank results**: Sorts by view count (higher confidence)
7. **Generate CSV**: Creates batch plan entries with top 4 URLs per angle

### Example: Generate Jab References

Scout for jab videos:
```powershell
python scout_youtube_by_golden_seeds.py --api-key $env:YOUTUBE_API_KEY --technique jab
```

This searches for:
- "boxing jab tutorial front view" (from Golden_Seeds/Jab/FrontJab_Front.MOV)
- "boxing jab tutorial side_right view" (from Golden_Seeds/Jab/FrontJab_Right.MOV)
- "how to throw a jab left 45 degree view" (from Golden_Seeds/Jab/FrontJab_45_Left.MOV)
- ... and more combinations

Merge results:
```powershell
python merge_scout_into_plan.py --scout-plan reference_poses/scout_batch_plan.csv --backup
```

Run batch collection:
```powershell
python run_reference_collection_batch.py
```

New references saved to: `reference_poses/jab/front_01.npy`, `jab/front_02.npy`, etc.

## 13) Useful output locations

- Main trainer output video: `output_demo.mp4`
- Pose-only video (if enabled): `datasets/pose_video.mp4`
- Preview videos: `reference_poses/previews/`
- Structured run artifacts: `data/runs/`

## 14) Oral expert-vs-program evaluation

1) Create your expert labels CSV from this template:

- `datasets/oral_eval/expert_scores_template.csv`

Required columns:
- `frame` (frame index where the expert judged the attempt)
- `expert_score_1_10` (oral score on 1-10 scale)

Optional columns:
- `sample_id`, `technique`, `track_id`, `notes`

2) Run matching + metrics:

```powershell
python DataCollectionScripts/evaluate_oral_scores.py    --metrics-csv data/runs/run_20260323_140007/metrics.csv    --expert-csv datasets/oral_eval/expert_scores_template.csv    --output-csv data/runs/oral_eval_matches.csv   --summary-json data/runs/oral_eval_summary.json    --max-frame-gap 20    --expert-pass-threshold 7.0
```

Outputs:
- `data/runs/oral_eval_matches.csv` (row-by-row expert vs program comparison)
- `data/runs/oral_eval_summary.json` (MAE, RMSE, Pearson r, pass agreement)