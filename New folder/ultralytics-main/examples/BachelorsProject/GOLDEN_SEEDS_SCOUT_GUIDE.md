# Golden Seeds YouTube Scout System

## Overview

The Golden Seeds YouTube Scout system automatically discovers quality YouTube videos for your reference collection batch process by using your existing Golden Seeds hand-recorded videos as templates.

### The Problem We're Solving

Previously, to build reference collections you had to:
1. Manually search YouTube for videos
2. Watch them to verify they show the right angle/technique
3. Fill in CSV rows with URLs manually
4. Hope the quality was good enough

Now, the system:
1. Automatically generates angle-specific search queries from Golden Seeds
2. Searches YouTube API for matching videos
3. Filters and ranks by quality metrics (views, definition, captions)
4. Auto-populates batch plan CSV entries
5. Ready to run batch collection immediately

## Components

### 1. Scout Utilities (`scout_utils.py`)

Helper functions for the scouting system:

- **`infer_angle_from_filename()`**: Analyzes video filename to determine camera angle
  - Patterns: "Front", "Right45", "SideLeft", "Behind", etc.
  - Returns: "front", "side_right", "left45", "behind", etc.

- **`generate_search_queries()`**: Creates YouTube search queries for a technique/angle
  - Uses configurable templates for each technique
  - Examples: "boxing jab tutorial side_right view", "karate roundhouse kick left 45 degree view"

- **`inventory_golden_seeds()`**: Scans `Golden_Seeds/` directory
  - Groups videos by technique and angle
  - Output: `{technique: {angle: [video_paths]}}`

- **`compute_pose_match_score()`**: Compares YouTube candidate against template (future enhancement)
  - Uses Dynamic Time Warping (DTW) distance between pose sequences
  - Scores 0-100 (higher = better match)

- **`create_csv_template_row()`**: Generates batch plan CSV row
  - Pre-fills technique, angle, source URLs
  - Sets command_ready="yes"

### 2. Main Scout Script (`scout_youtube_by_golden_seeds.py`)

The orchestrator that ties everything together:

**Input**:
- Golden Seeds videos in `reference_poses/Golden_Seeds/{technique}/{angle}.MOV`
- YouTube API key (environment variable or CLI arg)

**Process**:
1. Scan Golden Seeds for techniques/angles
2. For each target technique/angle:
   - Generate 5 search queries
   - Search YouTube (up to 2 pages per query)
   - Fetch detailed metrics (duration, definition, views, captions)
   - Filter by: duration (10-600s), views (>1000), definition (HD preferred)
   - Rank by view count (descending)

**Output**:
- `scout_candidates_golden_seeds.csv`: ALL candidates (for manual review)
- `scout_batch_plan.csv`: Top results (ready for batch collection)

**Key Arguments**:
```bash
--api-key API_KEY              # YouTube Data API v3 key (or env var)
--technique jab                # Limit to one technique (optional)
--angle side_right             # Limit to one angle (optional)
--max-candidates-per-angle 4   # Top N results per angle
--min-view-count 1000          # Filter by popularity
--min-duration-seconds 10      # Filter by video length
--max-duration-seconds 600     # Filter by video length
--dry-run                      # Show queries without API calls
```

### 3. Merge Helper (`merge_scout_into_plan.py`)

Intelligently merges scout results into your existing batch plan:

**Input**:
- `scout_batch_plan.csv` (from scout script)
- `generated_capture_plan_all_labels.csv` (existing plan)

**Merge Modes**:
- **append**: Add scout rows to existing plan (may create duplicates)
- **update** (default): Replace matches by (technique, angle), keep non-matching rows
- **replace**: Completely replace plan with scout results

**Safety Features**:
- `--backup`: Automatically backs up existing plan before writing
- `--dry-run`: Preview changes without writing
- Preserves user edits for rows not scouted

## Workflow Example: Scout Jab Videos

### Step 1: Scout YouTube

```powershell
python scout_youtube_by_golden_seeds.py --api-key YOUR_API_KEY --technique jab
```

**What happens**:
- Scans `reference_poses/Golden_Seeds/Jab/` and finds:
  - `FrontJab_Front.MOV` → angle "front"
  - `FrontJab_Right.MOV` → angle "side_right"
  - `FrontJab_45_Right.MOV` → angle "right45"
  - `FrontJab_Left.MOV` → angle "side_left"
  - `FrontJab_45_Left.MOV` → angle "left45"
  - `FrontJab_Right1.MOV` → angle "side_right" (duplicate)

- For each angle, generates queries like:
  - front: "boxing jab tutorial front view", "how to throw a jab front view", ...
  - side_right: "boxing jab tutorial side_right view", "jab technique right view", ...

- Searches YouTube for each query (up to 2 pages, 50 results per page)

- Fetches video details: duration, definition, view/like/comment counts, captions

- Filters:
  - Duration: 10-600 seconds ✓
  - Views: >1000 ✓
  - Definition: HD preferred

- Returns ~50-100 candidates

**Outputs**:
- `reference_poses/scout_candidates_golden_seeds.csv` (detailed for review)
- `reference_poses/scout_batch_plan.csv` (top 4 per angle, ready to use)

### Step 2: Review (Optional)

Open `scout_candidates_golden_seeds.csv` and check:
- Are the titles relevant? (e.g., contain "jab", "boxing", "tutorial")
- Are the channels reputable? (e.g., official boxing channels)
- Good view counts? (higher = more reliable)
- HD quality? (prefer 1080p)
- Do they have captions? (helps with pose extraction)

Remove any obviously bad candidates if needed.

### Step 3: Merge Into Batch Plan

```powershell
python merge_scout_into_plan.py `
  --scout-plan reference_poses/scout_batch_plan.csv `
  --existing-plan reference_poses/generated_capture_plan_all_labels.csv `
  --backup `
  --merge-mode update
```

**What happens**:
- Loads existing plan (has your manual entries)
- Loads scout results (new auto-found videos)
- For each technique/angle in scout results:
  - If it already exists in plan: replace with scout results
  - If it's new: add to plan
- Preserves any rows not touched by scouting (your manual edits)
- Backs up original plan to: `generated_capture_plan_all_labels.csv.backup_20260319_143521.csv`

### Step 4: Run Batch Collection (Existing Process)

```powershell
python run_reference_collection_batch.py
```

**What happens** (this is the existing batch collection):
- Reads your batch plan CSV
- For each row with `command_ready=yes`:
  - Downloads videos from `source_url_1`, `source_url_2`, `source_url_3`, `source_url_4`
  - Extracts the best 150-frame window showing the technique
  - Saves as `reference_poses/{technique}/{angle}_{example_num:02d}.npy`

**Results**:
- `reference_poses/jab/front_01.npy` (1st front jab reference)
- `reference_poses/jab/front_02.npy` (2nd front jab reference)
- `reference_poses/jab/side_right_01.npy` (1st right-side jab reference)
- etc.

These can now be used for real-time recognition with `action_recognition.py`.

## CSV Plan Structure

### Input Format (Scout Batch Plan)

```csv
technique,angle,reference_key,status,source_url,source_url_1,source_url_2,source_url_3,source_url_4,segment_start_s,segment_end_s,quality,notes,target_technique_key,record_reference_key,command_ready,command
jab,front,jab__front,ready,https://www.youtube.com/watch?v=ABC123,https://www.youtube.com/watch?v=ABC123,https://www.youtube.com/watch?v=DEF456,https://www.youtube.com/watch?v=GHI789,https://www.youtube.com/watch?v=JKL012,,,auto scout,auto from scout: 50 candidates found,jab,jab__front,yes,
```

### Key Columns

- **technique**: Technique name (e.g., "jab", "fighting_stance")
- **angle**: Camera angle (e.g., "front", "side_right", "left45")
- **source_url_1-4**: Up to 4 YouTube URLs to try
- **status**: "ready" means it's enabled
- **command_ready**: "yes" means run_reference_collection_batch.py will process it
- **quality**: Auto-filled with scout metadata
- **notes**: Auto-filled with candidate count found

### Output Files (After Batch Collection)

```
reference_poses/
├── jab/
│   ├── front_01.npy         (150 frames, 17 keypoints, 2+ coords each)
│   ├── front_02.npy
│   ├── side_right_01.npy
│   └── side_right_02.npy
├── fighting_stance/
│   ├── front_01.npy
│   └── side_left_01.npy
└── Golden_Seeds/
    ├── Jab/
    │   ├── FrontJab_Front.MOV
    │   └── FrontJab_Right.MOV
    └── FightingStance/
        └── FightingStance_Front.MOV
```

## API Key Setup

### Option 1: Environment Variable (Recommended)

```powershell
$env:YOUTUBE_API_KEY = "YOUR_API_KEY_HERE"
python scout_youtube_by_golden_seeds.py
```

### Option 2: Command Line

```powershell
python scout_youtube_by_golden_seeds.py --api-key "YOUR_API_KEY_HERE"
```

### Get a YouTube Data API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable YouTube Data API v3
4. Create Service Account or OAuth 2.0 credential
5. Copy API key to environment or CLI

**Rate Limits**:
- Free tier: 10,000 requests/day
- Scout for all techniques: ~50-100 requests
- Scout for one technique: ~5-10 requests

## Advanced Features & Future Enhancements

### Current Features

✓ Automatic angle inference from Golden Seeds filenames
✓ Angle-specific query generation
✓ YouTube API integration (search + details)
✓ Quality filtering (duration, views, definition, captions)
✓ CSV generation for batch collection
✓ Safe merging into existing plans with backups
✓ Dry-run mode for planning

### Planned Features

- **Pose-based matching**: Compare YouTube candidates against Golden Seeds using pose DTW distance
  - Would auto-score each candidate by pose similarity
  - Better filtering than just metadata

- **Stance detection**: Automatically classify orthodox vs southpaw stance

- **YouTube crawler**: Save video segments locally for offline analysis

- **Template library**: Store learned pose signatures for each technique/angle

- **Continuous scouting**: Periodic background jobs to keep reference library updated

- **Multi-angle extraction**: Extract multiple angles from single long video

## Troubleshooting

### "API key missing"
Set `YOUTUBE_API_KEY` environment variable or use `--api-key` argument.

### "No queries generated"
Check that Golden Seeds directory exists and contains video files with standard naming patterns (include "Front", "Right", "Left", "45", etc.).

### "No candidates found"
- Check query results manually on YouTube
- Try with higher `--min-view-count` or remove it
- Technique might not have enough YouTube content (very niche martial arts)

### CSV merge creates duplicates
Use `--merge-mode update` instead of `--merge-mode append` to avoid duplicates.

### Bad quality references extracted
Review `scout_candidates_golden_seeds.csv`:
- Candidates with <1000 views: Often low quality
- Candidates without HD: May be harder to extract poses
- Look for "Definition: hd" and "Caption: true" columns

## Integration with Existing System

```
┌─────────────────────────────────────────────────┐
│ GOLDEN SEEDS (Hand-Recorded Videos)             │
│ reference_poses/Golden_Seeds/Jab/*.MOV          │
│ └─ Use as TEMPLATES for search and validation   │
└─────────────────────────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │ SCOUT YOUTUBE                 │
        │ scout_youtube_by_golden_seeds │
        │ (generates queries, searches) │
        └───────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│ SCOUT RESULTS (CSV with YouTube URLs)           │
│ scout_batch_plan.csv / candidates CSV           │
│ └─ Filterable, reviewable                       │
└─────────────────────────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │ MERGE INTO BATCH PLAN         │
        │ merge_scout_into_plan.py      │
        │ (intelligent CSV merge)       │
        └───────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│ BATCH COLLECTION PLAN CSV                       │
│ generated_capture_plan_all_labels.csv           │
│ └─ Ready for batch collection                   │
└─────────────────────────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │ RUN BATCH COLLECTION          │
        │ run_reference_collection_batch│
        │ (downloads + extracts)        │
        └───────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│ EXTRACTED REFERENCES (Pose Keypoints)           │
│ reference_poses/jab/*.npy                       │
│ reference_poses/fighting_stance/*.npy           │
│ └─ Ready for real-time recognition              │
└─────────────────────────────────────────────────┘
```

## Development Notes

### Query Templates

Each technique has 5 query templates in `scout_utils.py`:

```python
TECHNIQUE_SEARCH_TEMPLATES = {
    "jab": [
        "boxing jab tutorial {angle}",
        "how to throw a jab {angle}",
        "jab technique {angle} view",
        "boxing jab form slow motion {angle}",
        "proper jab mechanics {angle}",
    ],
    # ... more techniques
}
```

Add more templates to improve search coverage:
```python
"jab": [
    "... existing templates ...",
    "professional jab technique {angle}",
    "boxing masterclass jab {angle}",
]
```

### Extending Pose Matching

The `compute_pose_match_score()` function in `scout_utils.py` is a foundation for future pose-based filtering:

```python
# Score 0-100: how similar is candidate to template?
score = compute_pose_match_score(candidate_pose, template_pose)

if score > 75:  # Good match
    candidates.append((video_url, score))
```

This could be integrated into the scout script to auto-score each candidate.

## Configuration Reference

Default parameters in `scout_youtube_by_golden_seeds.py`:

```python
# Discovery
--max-results-per-query 50         # YouTube returns max 50 per call
--max-pages-per-query 2            # 50*2 = 100 videos per query

# Filtering
--min-duration-seconds 10          # Action videos at least 10s
--max-duration-seconds 600         # No 10+ hour streams
--min-view-count 1000              # Popular enough to be good quality

# Ranking
--order relevance                  # Rank by relevance (alt: date, viewCount)

# Output
--max-candidates-per-angle 4       # Fill source_url_1-4 columns

# Quality
--ca-bundle ""                     # TLS certificate bundle (if needed)
--insecure-skip-tls-verify false  # Don't disable TLS in production
```

Adjust these based on your results. If too few candidates found, decrease `--min-view-count` or increase `--max-pages-per-query`.
