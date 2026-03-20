"""Utilities for scouting YouTube videos using Golden Seeds as templates.

This module provides functions to:
1. Infer technique/angle from Golden Seeds filenames
2. Generate YouTube search queries for each technique/angle
3. Match YouTube candidates against Golden Seeds using pose similarity
4. Generate CSV entries for batch collection
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def infer_angle_from_filename(file_name: str) -> str | None:
    """Infer viewing angle from filename pattern (same logic as in run_golden_seed_technique.py).
    
    Returns one of: front, left45, right45, side_left, side_right, side, behind, or None.
    """
    text = file_name.lower()

    if "behind" in text or "rear" in text or "back" in text:
        return "behind"
    if "left45" in text or "45left" in text or "45_left" in text or "left_45" in text:
        return "left45"
    if "right45" in text or "45right" in text or "45_right" in text or "right_45" in text:
        return "right45"
    if "side_left" in text or "left_side" in text:
        return "side_left"
    if "side_right" in text or "right_side" in text:
        return "side_right"
    if "front" in text:
        return "front"
    if re.search(r"(^|[_\-.\s])left($|[_\-.\s])", text):
        return "side_left"
    if re.search(r"(^|[_\-.\s])right($|[_\-.\s])", text):
        return "side_right"
    if "side" in text or "profile" in text:
        return "side"
    return None


def angle_to_camera_description(angle: str) -> str:
    """Convert angle code to descriptive text for YouTube search queries.
    
    Examples:
        front -> "front view"
        left45 -> "left 45 degree view"
        side_left -> "side/left view"
    """
    desc_map = {
        "front": "front view",
        "left45": "left 45 degree view",
        "right45": "right 45 degree view",
        "side_left": "side left view",
        "side_right": "side right view",
        "side": "side view",
        "behind": "back view",
    }
    return desc_map.get(angle, angle)


TECHNIQUE_SEARCH_TEMPLATES = {
    # Boxing techniques
    "jab": [
        "boxing jab tutorial {angle}",
        "how to throw a jab {angle}",
        "jab technique {angle} view",
        "boxing jab form slow motion {angle}",
        "proper jab mechanics {angle}",
    ],
    "cross": [
        "boxing cross punch tutorial {angle}",
        "how to throw a cross {angle}",
        "cross punch technique {angle} view",
        "boxing cross form {angle}",
        "power cross boxing {angle}",
    ],
    "hook": [
        "boxing hook punch tutorial {angle}",
        "how to throw a hook {angle}",
        "hook punch technique {angle} view",
        "boxing hook form {angle}",
        "power hook boxing {angle}",
    ],
    "uppercut": [
        "boxing uppercut tutorial {angle}",
        "how to throw an uppercut {angle}",
        "uppercut technique {angle} view",
        "boxing uppercut form {angle}",
        "power uppercut {angle}",
    ],
    # Kicking techniques
    "front kick": [
        "karate front kick tutorial {angle}",
        "how to throw a front kick {angle}",
        "front kick technique {angle} view",
        "martial arts front kick {angle}",
        "proper front kick form {angle}",
    ],
    "roundhouse kick": [
        "karate roundhouse kick tutorial {angle}",
        "how to throw a roundhouse kick {angle}",
        "roundhouse kick technique {angle} view",
        "martial arts roundhouse kick {angle}",
        "proper roundhouse kick form {angle}",
    ],
    "side kick": [
        "karate side kick tutorial {angle}",
        "how to throw a side kick {angle}",
        "side kick technique {angle} view",
        "martial arts side kick {angle}",
        "proper side kick form {angle}",
    ],
    # Stance/positioning
    "fighting stance": [
        "boxing fighting stance tutorial {angle}",
        "martial arts fighting stance {angle}",
        "fighting stance technique {angle} view",
        "proper boxing stance {angle}",
        "martial arts stance form {angle}",
    ],
}


def generate_search_queries(technique: str, angle: str, num_queries: int = 5) -> list[str]:
    """Generate YouTube search queries for a technique/angle combination.
    
    Args:
        technique: Technique name (e.g., "jab", "fighting_stance")
        angle: Angle code (e.g., "front", "side_right")
        num_queries: Maximum number of queries to return
    
    Returns:
        List of YouTube search query strings
    """
    technique_lower = technique.lower().replace("_", " ")
    angle_desc = angle_to_camera_description(angle)
    
    templates = TECHNIQUE_SEARCH_TEMPLATES.get(technique_lower, [])
    if not templates:
        # Fallback: generic queries
        templates = [
            f"{technique_lower} tutorial {angle_desc}",
            f"how to {technique_lower} {angle_desc}",
            f"{technique_lower} technique {angle_desc}",
        ]
    
    queries = []
    for template in templates[:num_queries]:
        query = template.format(angle=angle_desc).strip()
        if query not in queries:  # Avoid duplicates
            queries.append(query)
    
    return queries[:num_queries]


def inventory_golden_seeds(golden_seeds_dir: Path) -> dict[str, dict[str, list[Path]]]:
    """Scan Golden Seeds directory and organize by technique and angle.
    
    Returns:
        nested dict: {technique: {angle: [Path, ...]}}
    """
    inventory: dict[str, dict[str, list[Path]]] = {}
    
    if not golden_seeds_dir.exists():
        return inventory
    
    for technique_dir in golden_seeds_dir.iterdir():
        if not technique_dir.is_dir():
            continue
        
        technique = technique_dir.name.lower().replace("_", " ")
        inventory.setdefault(technique, {})
        
        for video_file in technique_dir.glob("*"):
            if video_file.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
                angle = infer_angle_from_filename(video_file.name)
                if angle:
                    inventory[technique].setdefault(angle, []).append(video_file)
    
    return inventory


def load_pose_from_npy(npy_path: Path) -> np.ndarray | None:
    """Load a pose sequence from a .npy file.
    
    Expected shape: (T, K, 2+) where T=frames, K=keypoints, >=2 coords.
    """
    try:
        data = np.load(str(npy_path))
        if data.ndim == 3 and data.shape[1] >= 10 and data.shape[2] >= 2:  # Basic validation
            return data.astype(np.float32)
    except Exception:
        pass
    return None


def frame_pose_distance(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Euclidean distance between two normalized pose frames (K,2)."""
    valid = np.isfinite(frame_a).all(axis=1) & np.isfinite(frame_b).all(axis=1)
    if not valid.any():
        return float("inf")
    return float(np.mean(np.linalg.norm(frame_a[valid] - frame_b[valid], axis=1)))


def dtw_pose_distance(seq_a: np.ndarray, seq_b: np.ndarray, max_dist: float = 500.0) -> float:
    """Dynamic Time Warping distance between pose sequences (T,K,2).
    
    Returns normalized distance (0-1 roughly, can be >1 for very different poses).
    Capped at max_dist to avoid computing full matrix for very dissimilar sequences.
    """
    if seq_a.ndim != 3 or seq_b.ndim != 3:
        return float("inf")
    
    na = seq_a.shape[0]
    nb = seq_b.shape[0]
    if na == 0 or nb == 0:
        return float("inf")
    
    dp = np.full((na + 1, nb + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = frame_pose_distance(seq_a[i - 1], seq_b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
            
            # Early termination if accumulated distance exceeds threshold
            if dp[i, j] > max_dist:
                return float("inf")
    
    normalized = float(dp[na, nb] / max(na, nb))
    return normalized


def compute_pose_match_score(
    candidate_seq: np.ndarray,
    template_seq: np.ndarray,
    dtw_weight: float = 0.7,
    normalization_weight: float = 0.3,
) -> float:
    """Compute match score between candidate and template pose sequences (0-100).
    
    Higher score = better match.
    
    Args:
        candidate_seq: Pose sequence from YouTube candidate (T,K,2+)
        template_seq: Ground truth from Golden Seeds (T,K,2+)
        dtw_weight: Weight for DTW distance score (0-1)
        normalization_weight: Weight for sequence length match (0-1)
    
    Returns:
        Score 0-100 (100 = perfect match)
    """
    if candidate_seq.ndim != 3 or template_seq.ndim != 3:
        return 0.0
    
    # DTW distance score (lower is better, normalized to 0-100)
    dtw_dist = dtw_pose_distance(candidate_seq, template_seq)
    dtw_score = max(0.0, 100.0 - dtw_dist * 50.0)  # Scale so 2.0 distance = 0 score
    
    # Sequence length match score (penalize if very different lengths)
    len_ratio = min(candidate_seq.shape[0], template_seq.shape[0]) / max(
        candidate_seq.shape[0], template_seq.shape[0]
    )
    len_score = len_ratio * 100.0
    
    combined_score = (dtw_score * dtw_weight + len_score * normalization_weight)
    return min(100.0, max(0.0, combined_score))


def create_csv_template_row(
    technique: str,
    angle: str,
    source_urls: list[str],
    target_technique_key: str | None = None,
    notes: str = "",
) -> dict[str, str]:
    """Create a CSV row template for the batch collection plan.
    
    Args:
        technique: Technique name (e.g., "jab")
        angle: Angle code (e.g., "side_right")
        source_urls: List of YouTube URLs (up to 4)
        target_technique_key: Normalized technique key; auto-generated if None
        notes: Optional notes about the source
    
    Returns:
        Dictionary matching generated_capture_plan_all_labels.csv column structure
    """
    target_tech = target_technique_key or technique.lower().replace(" ", "_")
    reference_key = f"{target_tech}__{angle}"
    
    # Pad source_urls to 4 columns
    padded_urls = (source_urls + [""] * 4)[:4]
    
    return {
        "technique": technique,
        "angle": angle,
        "reference_key": reference_key,
        "status": "ready",
        "source_url": padded_urls[0],
        "source_url_1": padded_urls[0],
        "source_url_2": padded_urls[1],
        "source_url_3": padded_urls[2],
        "source_url_4": padded_urls[3],
        "segment_start_s": "",
        "segment_end_s": "",
        "quality": "auto: from Golden Seeds scout",
        "notes": notes,
        "target_technique_key": target_tech,
        "record_reference_key": reference_key,
        "command_ready": "yes",
        "command": "",  # Will be generated by run_reference_collection_batch.py
    }


if __name__ == "__main__":
    # Quick test
    print("Scout Utils Module Loaded")
    
    # Test query generation
    print("\n=== Sample Queries ===")
    for technique in ["jab", "roundhouse kick", "fighting stance"]:
        for angle in ["front", "side_right"]:
            queries = generate_search_queries(technique, angle, num_queries=2)
            print(f"{technique} / {angle}:")
            for q in queries:
                print(f"  - {q}")
    
    # Test inventory
    print("\n=== Golden Seeds Inventory ===")
    project_root = Path(__file__).parent
    golden_dir = project_root / "reference_poses" / "Golden_Seeds"
    inv = inventory_golden_seeds(golden_dir)
    for tech, angles in sorted(inv.items()):
        print(f"{tech}:")
        for angle, files in sorted(angles.items()):
            print(f"  {angle}: {len(files)} file(s)")
