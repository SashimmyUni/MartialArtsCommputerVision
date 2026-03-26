from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _resolve_project_path(path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate PowerShell commands for recording reference poses from a reviewed candidate CSV."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="reference_poses/jab_video_candidates.csv",
        help="Input CSV produced by scrape_jab_candidates.py and manually reviewed.",
    )
    parser.add_argument(
        "--output-ps1",
        type=str,
        default="reference_poses/generated_capture_commands.ps1",
        help="Output PowerShell script file.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="reference_poses/generated_capture_plan.csv",
        help="Output CSV containing normalized keys and generated commands.",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default="python",
        help="Python executable to use in generated commands.",
    )
    parser.add_argument(
        "--runner-script",
        type=str,
        default="action_recognition.py",
        help="Path to action_recognition.py relative to where commands will run.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo26n-pose.pt",
        help="YOLO pose model weights argument.",
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="reference_poses",
        help="Reference directory argument for action_recognition.py.",
    )
    parser.add_argument(
        "--save-kpts-dir",
        type=str,
        default="keypoints",
        help="Keypoint output dir argument for action_recognition.py.",
    )
    parser.add_argument(
        "--num-video-sequence-samples",
        type=int,
        default=20,
        help="num_video_sequence_samples argument.",
    )
    parser.add_argument(
        "--skip-frame",
        type=int,
        default=1,
        help="skip_frame argument.",
    )
    parser.add_argument(
        "--reference-search-max-frames",
        type=int,
        default=0,
        help="maximum frames to search for a valid reference before skipping a clip; 0 disables the limit.",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include all rows, not only keep=yes rows.",
    )
    parser.add_argument(
        "--allow-missing-angle",
        action="store_true",
        help="Allow rows with empty angle. If empty, record key is just <technique>.",
    )
    parser.add_argument(
        "--ref-min-return-closure",
        type=float,
        default=0.20,
        help="Minimum return closure threshold.",
    )
    return parser.parse_args()


def _normalize_key(text: str) -> str:
    cleaned = (text or "").strip().lower().replace("-", " ")
    parts = [p for p in cleaned.split() if p]
    return "_".join(parts)


def _extract_video_id(source_url: str) -> str:
    parsed = urlparse(source_url)
    if parsed.hostname in {"youtu.be"}:
        return parsed.path.strip("/")
    qs = parse_qs(parsed.query)
    return (qs.get("v") or [""])[0]


def _slug_words(text: str, limit: int = 4) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    stop_words = {
        "the",
        "a",
        "an",
        "to",
        "how",
        "in",
        "of",
        "for",
        "and",
        "with",
        "by",
        "on",
        "your",
        "this",
        "that",
        "perfect",
        "boxing",
        "jab",
        "shorts",
        "tutorial",
        "technique",
    }
    words = [w for w in text.split() if w and w not in stop_words]
    return "_".join(words[:limit])


def _suggest_view_bucket(row: dict[str, str]) -> str:
    text = " ".join(
        [
            row.get("title", ""),
            row.get("tags", ""),
            row.get("notes", ""),
            row.get("channel_title", ""),
        ]
    ).lower()
    if any(token in text for token in ("southpaw", "lead hand", "lead jab", "right lead")):
        return "left45"
    if any(token in text for token in ("orthodox", "rear side", "right hand stance", "left lead")):
        return "right45"
    if any(token in text for token in ("footwork", "pivot", "stance", "body", "mechanics", "step", "hip")):
        return "lead_side"
    if any(token in text for token in ("behind", "from behind", "rear view", "back view")):
        return "behind"
    if any(token in text for token in ("right side", "right profile", "side right", "from right")):
        return "side_right"
    if any(token in text for token in ("left side", "left profile", "side left", "from left")):
        return "side_left"
    return "front"


def _suggest_angle_name(row: dict[str, str], used_angle_keys: set[str]) -> str:
    bucket = _suggest_view_bucket(row)
    title_slug = _slug_words(row.get("title", ""))
    channel_slug = _slug_words(row.get("channel_title", ""), limit=2)
    video_id = _extract_video_id(row.get("url", ""))[:6] or "clip"

    parts = [bucket]
    if title_slug:
        parts.append(title_slug)
    elif channel_slug:
        parts.append(channel_slug)
    parts.append(video_id)

    candidate = _normalize_key("_".join(parts))
    base = candidate or f"{bucket}_{video_id}"
    candidate = base
    suffix = 2
    while candidate in used_angle_keys:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_angle_keys.add(candidate)
    return candidate


def _truthy_keep(value: str) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y", "keep"}


def _ps_quote(s: str) -> str:
    # Escape embedded double quotes for PowerShell.
    return '"' + s.replace('"', '`"') + '"'


def build_command(
    python_exe: str,
    runner_script: str,
    weights: str,
    source_url: str,
    record_reference_key: str,
    target_technique_key: str,
    reference_dir: str,
    save_kpts_dir: str,
    num_video_sequence_samples: int,
    skip_frame: int,
    reference_search_max_frames: int,
    ref_min_return_closure: float,
) -> str:
    python_exe = (python_exe or "python").strip().strip('"')
    cmd_parts = [
        f"& {_ps_quote(python_exe)}",
        _ps_quote(runner_script),
        "--weights",
        _ps_quote(weights),
        "--source",
        _ps_quote(source_url),
        "--record-reference",
        _ps_quote(record_reference_key),
        "--target-technique",
        _ps_quote(target_technique_key),
        "--reference-dir",
        _ps_quote(reference_dir),
        "--num-video-sequence-samples",
        str(num_video_sequence_samples),
        "--skip-frame",
        str(skip_frame),
        "--save-kpts-dir",
        _ps_quote(save_kpts_dir),
        "--record-reference-max-saves",
        "0",
        "--reference-capture-cooldown-frames",
        "24",
        "--disable-video-classifier",
        "--visualize-pose",
        "--person-selection-mode",
        _ps_quote("most_motion"),
        "--ref-min-return-closure",
        f"{float(ref_min_return_closure):.2f}",
    ]
    if reference_search_max_frames > 0:
        cmd_parts.extend([
            "--reference-search-max-frames",
            str(reference_search_max_frames),
        ])
    return " ".join(cmd_parts)


def main() -> int:
    args = parse_args()

    in_path = _resolve_project_path(args.input_csv)
    if not in_path.exists():
        print(f"error: input CSV not found: {in_path}")
        return 2

    rows_out: list[dict[str, str]] = []
    used_angle_keys: set[str] = set()

    with in_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=2):
            keep_val = row.get("keep", "")
            if not args.include_all and not _truthy_keep(keep_val):
                continue

            source_url = (row.get("url", "") or "").strip()
            if not source_url:
                continue

            technique = _normalize_key(row.get("technique", ""))
            angle = _normalize_key(row.get("angle", ""))

            if not technique:
                continue
            if not angle:
                if args.allow_missing_angle:
                    angle = _suggest_angle_name(row, used_angle_keys)
                else:
                    continue
            else:
                used_angle_keys.add(angle)

            record_reference_key = f"{technique}__{angle}" if angle else technique
            target_technique_key = technique

            command = build_command(
                python_exe=args.python_exe,
                runner_script=args.runner_script,
                weights=args.weights,
                source_url=source_url,
                record_reference_key=record_reference_key,
                target_technique_key=target_technique_key,
                reference_dir=args.reference_dir,
                save_kpts_dir=args.save_kpts_dir,
                num_video_sequence_samples=args.num_video_sequence_samples,
                skip_frame=args.skip_frame,
                reference_search_max_frames=args.reference_search_max_frames,
                ref_min_return_closure=args.ref_min_return_closure,
            )

            rows_out.append(
                {
                    "line_number": str(idx),
                    "keep": keep_val,
                    "technique": row.get("technique", ""),
                    "angle": angle,
                    "technique_key": technique,
                    "angle_key": angle,
                    "record_reference_key": record_reference_key,
                    "target_technique_key": target_technique_key,
                    "url": source_url,
                    "title": row.get("title", ""),
                    "segment_start_s": row.get("segment_start_s", ""),
                    "segment_end_s": row.get("segment_end_s", ""),
                    "command": command,
                }
            )

    out_ps1 = _resolve_project_path(args.output_ps1)
    out_ps1.parent.mkdir(parents=True, exist_ok=True)

    ps1_lines = [
        "# Auto-generated reference capture commands",
        "# Run from examples/BachelorsProject",
        "# Execute one command at a time, then press 'q' in the video window when the reference is captured.",
        "",
    ]

    for i, r in enumerate(rows_out, start=1):
        ps1_lines.append(f"# {i}. {r['record_reference_key']} | {r['title']}")
        ps1_lines.append(r["command"])
        ps1_lines.append("")

    out_ps1.write_text("\n".join(ps1_lines), encoding="utf-8")

    out_csv = _resolve_project_path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "line_number",
            "keep",
            "technique",
            "angle",
            "technique_key",
            "angle_key",
            "record_reference_key",
            "target_technique_key",
            "url",
            "title",
            "segment_start_s",
            "segment_end_s",
            "command",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"generated {len(rows_out)} command(s)")
    print(f"powerShell script: {out_ps1}")
    print(f"capture plan csv: {out_csv}")
    if len(rows_out) == 0:
        print("note: no rows matched filters. Check keep column or use --include-all.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
