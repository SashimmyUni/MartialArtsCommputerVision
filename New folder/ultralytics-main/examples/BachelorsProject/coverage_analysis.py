#!/usr/bin/env python3
import csv
import os
from pathlib import Path
from collections import defaultdict

ref_dir = Path(__file__).parent / "reference_poses"
plan_csv = ref_dir / "generated_capture_plan_all_labels.csv"

# Load plan
plan_rows = []
with open(plan_csv, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        plan_rows.append(row)

# Count existing .npy files per (technique, angle)
existing = defaultdict(int)
for npy_file in ref_dir.glob("*/[!_]*.npy"):
    technique = npy_file.parent.name
    angle = npy_file.stem.rsplit("_", 1)[0]  # Remove _01, _02 suffix
    key = f"{technique}__{angle}"
    existing[key] += 1

# Analyze by status
print("=" * 80)
print("COVERAGE ANALYSIS")
print("=" * 80)

ready_rows = [r for r in plan_rows if r.get("status") == "ready"]
print(f"\nReadiness: {len(ready_rows)}/{len(plan_rows)} rows marked 'ready'")

# Per-ready-row analysis
missing_ready = []
low_ready = []
good_ready = []

for row in ready_rows:
    tech = row["technique"]
    angle = row["angle"]
    key = f"{tech}__{angle}"
    count = existing.get(key, 0)
    
    if count == 0:
        missing_ready.append((tech, angle, key))
    elif count < 4:
        low_ready.append((tech, angle, key, count))
    else:
        good_ready.append((tech, angle, key, count))

print(f"\nReady-plan breakdown:")
print(f"  ✓ With ≥4 samples: {len(good_ready)}")
print(f"  ⚠ With 1-3 samples: {len(low_ready)}")
print(f"  ✗ With 0 samples: {len(missing_ready)}")

if missing_ready:
    print(f"\nMissing (0 samples) - {len(missing_ready)} ready-keys:")
    for tech, angle, key in sorted(missing_ready):
        sources = [row.get(f"source_url_{i}") for row in ready_rows if row["technique"]==tech and row["angle"]==angle for i in range(1,5) if row.get(f"source_url_{i}")]
        print(f"  {key:30} sources: {len(sources)}")

if low_ready:
    print(f"\nLow samples (1-3) - {len(low_ready)} ready-keys:")
    for tech, angle, key, count in sorted(low_ready, key=lambda x: x[3]):
        print(f"  {key:30} {count}/4 samples")

# All techniques coverage
print(f"\nAll-technique total coverage:")
all_keys = set()
for row in plan_rows:
    key = f"{row['technique']}__{row['angle']}"
    all_keys.add(key)

covered = sum(1 for k in all_keys if existing.get(k, 0) > 0)
print(f"  {covered}/{len(all_keys)} keys with ≥1 sample")
print(f"  {sum(1 for k in all_keys if existing.get(k, 0) >= 4)}/{len(all_keys)} keys with ≥4 samples")

# Generate batch collection command for gaps
print("\n" + "=" * 80)
print("BATCH COLLECTION TARGETS")
print("=" * 80)

targets = missing_ready + low_ready
if targets:
    print(f"\nGenerate batch collection for {len(targets)} gap targets:")
    print(f"  run_reference_collection_batch.py --examples-per-angle 4")
    print(f"\nThis will collect:")
    print(f"  - All {len(missing_ready)} missing keys (4 each)")
    print(f"  - Top-up {len(low_ready)} low-sample keys to 4 each")
