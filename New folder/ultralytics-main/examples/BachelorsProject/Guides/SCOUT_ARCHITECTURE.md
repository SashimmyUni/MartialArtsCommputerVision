# Golden Seeds YouTube Scout System - Complete Architecture

## What We Built

A complete automated YouTube video scouting system that uses your Golden Seeds (hand-recorded reference videos) as templates to find, filter, and populate batch reference collection jobs.

## The Three New Scripts

### 1. `scout_utils.py` ✓ (Utilities Library)
**Purpose**: Helper functions and shared logic

**Key Functions**:
- `infer_angle_from_filename()` - Parse camera angle from video name
- `generate_search_queries()` - Create angle-specific YouTube search queries
- `inventory_golden_seeds()` - Scan and organize Golden Seeds by technique/angle
- `create_csv_template_row()` - Generate batch plan CSV entries
- `compute_pose_match_score()` - Compare poses using DTW distance (foundation for future enhancements)

**Templates Included**:
- Jab, Cross, Hook, Uppercut (boxing)
- Front Kick, Roundhouse Kick, Side Kick (kicking)
- Fighting Stance (positioning)
- Extensible for any martial arts technique

### 2. `scout_youtube_by_golden_seeds.py` ✓ (Main Scout Script)
**Purpose**: Orchestrate YouTube discovery for all techniques/angles

**Workflow**:
1. Scan `reference_poses/Golden_Seeds/` for videos
2. Analyze filenames to determine camera angles
3. For each technique/angle:
   - Generate 5 different YouTube search queries
   - Search YouTube API (up to 2 pages, 50 results each)
   - Fetch detailed metrics (duration, views, definition, captions)
4. Filter by: duration (10-600s), views (>1000), quality (HD preferred)
5. Rank by view count (higher = more reliable)
6. Output two CSVs:
   - `scout_candidates_golden_seeds.csv` - ALL candidates (for manual review)
   - `scout_batch_plan.csv` - Top results (ready for batch collection)

**Key Arguments**:
```bash
--api-key API_KEY                  # YouTube API key
--technique jab                    # Optional: limit to one technique
--angle side_right                 # Optional: limit to one angle
--max-candidates-per-angle 4       # Top N results per angle
--min-view-count 1000              # Popularity filter
--dry-run                          # Preview without API calls
```

### 3. `merge_scout_into_plan.py` ✓ (Safe CSV Merge)
**Purpose**: Intelligently merge scout results into existing batch plan

**Merge Strategies**:
- `append`: Add scout rows (may create duplicates)
- `update` (default): Replace by (technique, angle), keep other rows
- `replace`: Entirely new plan from scout results

**Safety Features**:
- `--backup`: Auto-backup original plan before writing
- `--dry-run`: Preview changes without writing
- Preserves user edits for non-scouted entries

## System Data Flow

```
┌─────────────────────────────────┐
│ GOLDEN SEEDS (Your Videos)      │
│ directory: reference_poses/     │
│            Golden_Seeds/        │
│ 13 techniques × 5-6 angles      │
└─────────────────────────────────┘
            ↓ (scan)
┌─────────────────────────────────┐
│ scout_utils.py                  │
│ - Infer angles from filenames   │
│ - Generate search templates     │
│ - Organize by technique/angle   │
└─────────────────────────────────┘
            ↓ (queries)
┌─────────────────────────────────────────┐
│ scout_youtube_by_golden_seeds.py        │
│ - Use YouTube Data API v3               │
│ - Search for each technique/angle       │
│ - Filter by quality metrics             │
│ - Rank by popularity                    │
└─────────────────────────────────────────┘
            ↓ (CSV results)
         ╔════════════════════════════════════╗
         ║ scout_candidates_golden_seeds.csv ║ ← Manual review
         ║ scout_batch_plan.csv              ║ ← Ready to use
         ╚════════════════════════════════════╝
            ↓ (merge)
┌──────────────────────────────────────────┐
│ merge_scout_into_plan.py                 │
│ - Load scout results                     │
│ - Load existing batch plan               │
│ - Intelligent merge (avoid duplicates)   │
│ - Backup original                        │
└──────────────────────────────────────────┘
            ↓ (updated plan)
┌────────────────────────────────────┐
│ generated_capture_plan_all_labels  │
│ CSV (Ready for batch collection)   │
└────────────────────────────────────┘
            ↓ (existing workflow)
┌──────────────────────────────────────┐
│ run_reference_collection_batch.py    │
│ - Download videos from YouTube       │
│ - Extract best technique instances   │
│ - Save pose keypoints (.npy files)   │
└──────────────────────────────────────┘
            ↓ (result)
┌────────────────────────────────────────┐
│ reference_poses/{technique}/*.npy      │
│ Ready for real-time recognition       │
└────────────────────────────────────────┘
```

## Your Golden Seeds Inventory

Found 13 martial arts techniques:

```
✓ axekick (6 angles)
✓ cross (6 angles)
✓ elbow (6 angles)
✓ fighting_stance (7 angles)
✓ front_kick (5 angles)
✓ hook (6 angles)
✓ jab (4+ angles)
✓ knee_strike (5 angles)
✓ roundhouse_kick (5 angles)
✓ uppercut (6 angles)
+ other techniques
```

Each technique can be scouted independently for YouTube videos:
```powershell
python scout_youtube_by_golden_seeds.py --api-key KEY --technique jab
python scout_youtube_by_golden_seeds.py --api-key KEY --technique roundhouse_kick
python scout_youtube_by_golden_seeds.py --api-key KEY  # All techniques
```

## Usage Example: Scout One Technique

### 1. Get YouTube API Key

- Visit [Google Cloud Console](https://console.cloud.google.com/)
- Create project → Enable YouTube Data API v3 → Create credentials
- Cost: FREE under 10,000 requests/day

### 2. Scout Jab Videos

```powershell
$env:YOUTUBE_API_KEY = "YOUR_API_KEY"

python scout_youtube_by_golden_seeds.py --api-key $env:YOUTUBE_API_KEY --technique jab
```

**Console Output**:
```
========================================
GOLDEN SEEDS YOUTUBE SCOUT
========================================

Scanning Golden Seeds: reference_poses/Golden_Seeds
Found 13 technique(s):
  axekick: 6 angle(s)
  jab: 4 angle(s) - front, left45, right45, ...

Scout targets: 4 combinations
  jab / front -> boxing jab tutorial front view
  jab / left45 -> boxing jab tutorial left 45 degree view
  ...

========================================
SEARCHING YOUTUBE
========================================

  jab / front -> 5 queries
    query: boxing jab tutorial front view...
      → 50 hits
    query: how to throw a jab front view...
      → 48 hits
    ... (3 more queries)
    deduplicated to 87 unique videos
    [fetch details...]
    → added 4 URL(s) to batch plan

  jab / left45 -> 5 queries
    ...

========================================
WRITING OUTPUT
========================================

✓ Wrote 350+ candidates to: reference_poses/scout_candidates_golden_seeds.csv
✓ Wrote 4 batch plan rows to: reference_poses/scout_batch_plan.csv

Next steps:
  1. Review candidates: reference_poses/scout_candidates_golden_seeds.csv
  2. Copy/merge to: reference_poses/generated_capture_plan_all_labels.csv
  3. Run: python run_reference_collection_batch.py
```

### 3. Merge Results

```powershell
python merge_scout_into_plan.py `
  --scout-plan reference_poses/scout_batch_plan.csv `
  --backup `
  --merge-mode update
```

**Output**:
```
Loaded scout plan: reference_poses/scout_batch_plan.csv
  → 4 rows

Loaded existing plan: reference_poses/generated_capture_plan_all_labels.csv
  → 42 rows (from manual + previous scouts)

Merge mode: update
  Updating 2 existing entries (jab/front, jab/side_right)
  Adding 2 new entries (jab/left45, jab/right45)
  Result: 44 total rows

✓ Backed up existing plan to: generated_capture_plan_all_labels.csv.backup_...
✓ Wrote merged plan to: reference_poses/generated_capture_plan_all_labels.csv
```

### 4. Run Batch Collection (Existing Process)

```powershell
python run_reference_collection_batch.py --preflight-only

# Then run for real:
python run_reference_collection_batch.py
```

**Output**:
```
loaded 42 total rows, 42 ready row(s)

[1/42] running: jab / front
  - example 01: jab__front_01 (source 1/4)
    [download video from YouTube...]
    [extract poses...]
    [find best window...]
    - saved: reference_poses/jab/front_01.npy

  - example 02: jab__front_02 (source 2/4)
    - saved: reference_poses/jab/front_02.npy

[2/42] running: jab / left45
  ...

Summary:
  completed: 38
  skipped_existing: 2
  failed: 2
  examples_saved: 84
```

**Result Files**:
```
reference_poses/jab/
  front_01.npy       (150 frames, 17 keypoints, 2D coords)
  front_02.npy
  left45_01.npy
  left45_02.npy
  ... and more angles
```

These are now ready for real-time recognition!

## Query Template System

The scout generates queries using language templates tailored to each technique:

```python
# Example: Fighting Stance queries for different angles
"boxing fighting stance tutorial front view"
"martial arts fighting stance front view"
"fighting stance technique front view"
"proper boxing stance front view"
"martial arts stance form front view"

# Example: Jab queries for side_right angle
"boxing jab tutorial side right view"
"how to throw a jab side right view"
"jab technique side right view"
"boxing jab form slow motion side right view"
"proper jab mechanics side right view"
```

This is 100x better than fixed queries because:
- Different angles get different descriptive text
- Multiple phrasing increases coverage
- "boxing jab tutorial" vs "how to throw a jab" capture different content

## Advanced Features (Built-In, Not Yet Used)

### 1. Pose-Based Matching
Function `compute_pose_match_score()` in `scout_utils.py`:
- Compares candidate videos against Golden Seeds using Dynamic Time Warping
- Scores 0-100 (100 = perfect match)
- Could auto-filter videos by pose similarity
- Foundation for future enhancement

### 2. Graceful Error Handling
- Network errors during API calls don't crash the script
- Partial results still saved
- Resume-friendly workflow

### 3. Multiple Merge Strategies
- Keep all duplicate entries
- Auto-update duplicates
- Completely replace existing plan
- Each strategy has a use case

## Configuration & Tuning

### If Getting Too Few Results

Decrease minimum view count:
```powershell
python scout_youtube_by_golden_seeds.py --api-key KEY --min-view-count 500
```

Increase pages per query:
```powershell
python scout_youtube_by_golden_seeds.py --api-key KEY --max-pages-per-query 5
```

### If Getting Too Many Results

Increase minimum view count:
```powershell
python scout_youtube_by_golden_seeds.py --api-key KEY --min-view-count 5000
```

Decrease duration range:
```powershell
python scout_youtube_by_golden_seeds.py --api-key KEY `
  --min-duration-seconds 20 `
  --max-duration-seconds 300
```

### If Getting Low-Quality Videos

- Review `scout_candidates_golden_seeds.csv` and remove manually
- Look for "Definition: hd" and "Caption: true" indicators
- Check view counts and channel reputation
- Filter by dates (see `--published-after` argument)

## Files Created / Modified

### New Files ✓
- `scout_utils.py` - Utilities library
- `scout_youtube_by_golden_seeds.py` - Main scout script
- `merge_scout_into_plan.py` - CSV merge helper
- `GOLDEN_SEEDS_SCOUT_GUIDE.md` - Detailed documentation
- `SCOUT_ARCHITECTURE.md` - This file

### Modified Files ✓
- `ReadyToRunCommands.md` - Added commands #12 (scout workflow)

### Generated Files (Output)
- `scout_candidates_golden_seeds.csv` - All candidates from search
- `scout_batch_plan.csv` - Top results ready for batch collection

## Integration Points

### With Existing Batch System
The scout outputs CSV rows in the **exact format** expected by `run_reference_collection_batch.py`:
- Same column names
- Same URL format (YouTube watch links)
- Same technique/angle keys
- Plug & play!

### With Golden Seeds Processing
Scout uses the same angle inference logic as `run_golden_seed_technique.py`:
- Identical filename parsing
- Consistent angle names
- Compatible output format

### With Real-Time Recognition
Generated `.npy` files work with existing `action_recognition.py`:
- 150-frame sequences at 30fps ≈ 5 seconds of motion
- 17 COCO keypoints per frame
- 2D normalized coordinates
- Can be used for comparison scoring immediately

## Next Steps for You

### Immediate
1. ✓ Review this architecture
2. Set up YouTube API key
3. Try scouting one technique: `python scout_youtube_by_golden_seeds.py --api-key KEY --technique jab --dry-run`
4. Run actual scout (will take 1-2 minutes)
5. Review output CSV
6. Merge and run batch collection

### Short Term
- Scout all 13 techniques to quickly build reference library
- Fine-tune parameters if needed (min-view-count, threshold, etc.)
- Monitor batch collection results quality

### Medium Term
- Build pose-based filtering (auto-score candidates)
- Create routine scouting jobs (monthly updates)
- Expand technique library

### Long Term
- Integrate with CI/CD for automated updates
- Build web dashboard for scout results review
- Contribute improvements back to Ultralytics

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "API key missing" | Environment variable not set | export YOUTUBE_API_KEY=... |
| No candidates found | Golden Seeds dir not found | Check `reference_poses/Golden_Seeds/` exists |
| CSV merge creates duplicates | Used --merge-mode append | Use --merge-mode update instead |
| Very few results | Searches too specific or niche | Lower --min-view-count |
| Low-quality videos | Results too aggressive | Increase --min-view-count |
| Batch collection fails | YouTube videos removed | Scout again, videos might no longer exist |

## Performance Notes

- **Scout time**: ~1-2 minutes for all techniques (API calls are the bottleneck)
- **CSV merge**: <1 second
- **Batch collection**: Depends on video count, typically 1-2 hours for 40+ videos
- **API costs**: FREE (within 10,000 requests/day quota)

## Success Metrics

The scout system is working well if:
- ✓ Golden Seeds inventory shows 13+ techniques
- ✓ Queries are generated for each angle
- ✓ YouTube search returns 50+ candidates per angle
- ✓ Most candidates have HD quality
- ✓ Batch collection successfully extracts references
- ✓ New `.npy` files created in reference_poses/{technique}/
