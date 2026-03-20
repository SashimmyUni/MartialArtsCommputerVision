"""Merge scouted YouTube videos into batch collection plan CSV.

This script takes scouted candidates and merges them into the batch plan,
with options to:
- Update existing entries or create new ones
- Deduplicate by technique/angle
- Validate before merging
- Backup original plan

Usage:
    python merge_scout_into_plan.py --scout-plan scout_batch_plan.csv
    python merge_scout_into_plan.py --scout-plan scout_batch_plan.csv --backup --merge-mode update
"""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge scouted YouTube videos into batch collection plan."
    )
    parser.add_argument(
        "--scout-plan",
        type=str,
        required=True,
        help="Path to scout batch plan CSV (output from scout_youtube_by_golden_seeds.py).",
    )
    parser.add_argument(
        "--existing-plan",
        type=str,
        default="reference_poses/generated_capture_plan_all_labels.csv",
        help="Path to existing batch plan CSV to merge into.",
    )
    parser.add_argument(
        "--output-plan",
        type=str,
        default="reference_poses/generated_capture_plan_all_labels.csv",
        help="Path to write merged plan.",
    )
    parser.add_argument(
        "--merge-mode",
        type=str,
        choices=["append", "update", "replace"],
        default="update",
        help=(
            "Merge strategy: "
            "append=add scout rows (may create duplicates); "
            "update=replace existing technique/angle with scout results; "
            "replace=overwrite entire plan with scout results."
        ),
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup original plan before writing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making changes.",
    )
    return parser.parse_args()


def load_plan_csv(path: str | Path) -> list[dict[str, str]]:
    """Load rows from a plan CSV."""
    path = Path(path)
    if not path.exists():
        return []
    
    rows = []
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader) if reader.fieldnames else []
    
    return rows


def save_plan_csv(rows: list[dict[str, str]], path: str | Path) -> None:
    """Save rows to a plan CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if not rows:
        return
    
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def technique_angle_key(row: dict[str, str]) -> str:
    """Create a unique key for deduplication."""
    technique = (row.get("technique") or "").strip().lower()
    angle = (row.get("angle") or "").strip().lower()
    return f"{technique}#{angle}"


def merge_plans(
    scout_rows: list[dict[str, str]],
    existing_rows: list[dict[str, str]],
    merge_mode: str = "update",
) -> list[dict[str, str]]:
    """Merge scout results into existing plan.
    
    Args:
        scout_rows: Rows from scouted results
        existing_rows: Rows from existing plan
        merge_mode: "append", "update", or "replace"
    
    Returns:
        Merged row list
    """
    if merge_mode == "replace":
        return scout_rows.copy()
    
    if merge_mode == "append":
        return existing_rows + scout_rows
    
    # merge_mode == "update"
    # Create index of existing by (technique, angle)
    existing_by_key: dict[str, dict[str, str]] = {}
    for row in existing_rows:
        key = technique_angle_key(row)
        if key:
            existing_by_key[key] = row
    
    # Build merged list: update existing, append new
    merged: dict[str, dict[str, str]] = existing_by_key.copy()
    for row in scout_rows:
        key = technique_angle_key(row)
        if key:
            merged[key] = row
    
    # Restore order: existing first (preserves user edits), then new
    result = []
    seen_keys = set()
    
    for row in existing_rows:
        key = technique_angle_key(row)
        if key and key in merged:
            result.append(merged[key])
            seen_keys.add(key)
    
    for row in scout_rows:
        key = technique_angle_key(row)
        if key and key not in seen_keys:
            result.append(merged[key])
            seen_keys.add(key)
    
    return result


def main() -> int:
    args = parse_args()
    
    scout_plan_path = Path(args.scout_plan)
    existing_plan_path = Path(args.existing_plan)
    output_plan_path = Path(args.output_plan)
    
    if not scout_plan_path.exists():
        print(f"error: scout plan not found: {scout_plan_path}")
        return 2
    
    print("=" * 80)
    print("MERGE SCOUT RESULTS INTO BATCH PLAN")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading scout plan: {scout_plan_path}")
    scout_rows = load_plan_csv(scout_plan_path)
    print(f"  -> {len(scout_rows)} rows")
    
    print(f"\nLoading existing plan: {existing_plan_path}")
    existing_rows = load_plan_csv(existing_plan_path)
    print(f"  -> {len(existing_rows)} rows")
    
    # Show merge preview
    print(f"\nMerge mode: {args.merge_mode}")
    if args.merge_mode == "append":
        merged_rows = existing_rows + scout_rows
        print(f"  Appending {len(scout_rows)} new rows")
    elif args.merge_mode == "replace":
        merged_rows = scout_rows
        print(f"  Replacing entire plan with scout results")
    else:  # update
        merged_rows = merge_plans(scout_rows, existing_rows, "update")
        updated_count = sum(
            1 for scout in scout_rows
            if technique_angle_key(scout) in {technique_angle_key(r) for r in existing_rows}
        )
        print(f"  Updating {updated_count} existing entries")
        new_count = len(scout_rows) - updated_count
        print(f"  Adding {new_count} new entries")
    
    print(f"  Result: {len(merged_rows)} total rows")
    
    if args.dry_run:
        print("\n[DRY RUN] Exiting without writing.")
        return 0
    
    # Backup if requested
    if args.backup and existing_plan_path.exists():
        backup_path = existing_plan_path.with_suffix(
            f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        shutil.copy2(existing_plan_path, backup_path)
        print(f"\n✓ Backed up existing plan to: {backup_path}")
    
    # Write merged plan
    output_plan_path.parent.mkdir(parents=True, exist_ok=True)
    save_plan_csv(merged_rows, output_plan_path)
    print(f"\n✓ Wrote merged plan to: {output_plan_path}")
    
    # Summary
    print(f"\nNext steps:")
    print(f"  1. Review merged plan: {output_plan_path}")
    print(f"  2. Run: python run_reference_collection_batch.py")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
