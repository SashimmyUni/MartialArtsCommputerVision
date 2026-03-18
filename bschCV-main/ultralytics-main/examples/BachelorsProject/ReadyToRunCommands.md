# Ready To Run Commands

Use these commands from this folder:

- `ultralytics-main/examples/BachelorsProject`

## 1) Optional environment setup (Windows PowerShell)

```powershell
cd "c:\Users\Sashi\Documents\Bachelor\Finalbsc\bschCV\ultralytics-main\examples\BachelorsProject"

# If your venv is at repo root:
& "c:\Users\Sashi\Documents\Bachelor\Finalbsc\bschCV\.venv\Scripts\Activate.ps1"
```

## 2) Fast trainer run on local video (recommended)

```powershell
python action_recognition.py --source 0 --target-technique jab --reference-dir reference_poses --output-path output_demo_Live.mp4 --skip-frame 2
```

## 3) Trainer run with pose visualization video output

```powershell
python action_recognition.py --source 0 --target-technique jab --reference-dir reference_poses --output-path output_demo.mp4 --disable-video-classifier --visualize-pose --pose-output-path datasets/pose_video.mp4 --skip-frame 2 --no-display
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
  --source "reference_poses/Golden_Seeds/jab/FrontJab_Right.MOV" `
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
python visualize_reference_pose.py --reference-file reference_poses/jab/frontjab_right_bestwindow_150f.npy --save-video reference_poses/previews/jab/frontjab_right_bestwindow_150f.mp4 --no-window --overwrite
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

## 10) Jab seed-gated batch helper script

```powershell
powershell -ExecutionPolicy Bypass -File .\run_jab_seed_batch.ps1 -ExamplesPerAngle 2 -NumVideoSequenceSamples 150 -RefMinScoreGate 55 -ReferenceSearchMaxFrames 260
```

## 11) Useful output locations

- Main trainer output video: `output_demo.mp4`
- Pose-only video (if enabled): `datasets/pose_video.mp4`
- Preview videos: `reference_poses/previews/`
- Structured run artifacts: `data/runs/`
