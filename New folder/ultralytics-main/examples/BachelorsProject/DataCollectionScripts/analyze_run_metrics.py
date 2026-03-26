from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _resolve_project_path(path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _as_float(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


@dataclass
class RunSummary:
    run_id: str
    metrics_path: str
    source: str
    rows: int
    correct: int
    accuracy_pct: float
    score_mean: float
    score_std: float
    score_min: float
    score_max: float
    threshold_mean: float
    cosine_mean: float
    dtw_mean: float
    angle_error_mean: float
    mean_pose_distance_mean: float
    top_reference_angle: str
    top_reference_angle_count: int
    top_feedback_1: str
    top_feedback_1_count: int
    top_feedback_2: str
    top_feedback_2_count: int


def _mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def _std(values: list[float], mean_value: float) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    var = sum((v - mean_value) ** 2 for v in finite) / len(finite)
    return math.sqrt(var)


def _safe_min(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return min(finite) if finite else float("nan")


def _safe_max(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return max(finite) if finite else float("nan")


def summarize_metrics_csv(metrics_path: Path) -> RunSummary | None:
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    run_id = rows[0].get("run_id", metrics_path.parent.name)
    sources = [r.get("source", "") for r in rows if (r.get("source", "") or "").strip()]
    source = sources[0] if sources else ""

    correct = sum(1 for r in rows if _as_bool(r.get("is_correct", "")))
    total = len(rows)

    scores = [_as_float(r.get("score", "")) for r in rows]
    thresholds = [_as_float(r.get("score_threshold", "")) for r in rows]
    cosine = [_as_float(r.get("cosine_similarity", "")) for r in rows]
    dtw = [_as_float(r.get("dtw_distance", "")) for r in rows]
    angle_err = [_as_float(r.get("angle_error", "")) for r in rows]
    pose_dist = [_as_float(r.get("mean_pose_distance", "")) for r in rows]

    score_mean = _mean(scores)
    score_std = _std(scores, score_mean) if math.isfinite(score_mean) else float("nan")

    angles = Counter((r.get("reference_angle", "") or "").strip() for r in rows)
    angles.pop("", None)
    top_angle, top_angle_count = ("", 0)
    if angles:
        top_angle, top_angle_count = angles.most_common(1)[0]

    fb1 = Counter((r.get("feedback_1", "") or "").strip() for r in rows)
    fb2 = Counter((r.get("feedback_2", "") or "").strip() for r in rows)
    fb1.pop("", None)
    fb2.pop("", None)
    top_fb1, top_fb1_count = ("", 0)
    top_fb2, top_fb2_count = ("", 0)
    if fb1:
        top_fb1, top_fb1_count = fb1.most_common(1)[0]
    if fb2:
        top_fb2, top_fb2_count = fb2.most_common(1)[0]

    return RunSummary(
        run_id=run_id,
        metrics_path=str(metrics_path),
        source=source,
        rows=total,
        correct=correct,
        accuracy_pct=(100.0 * correct / total) if total else float("nan"),
        score_mean=score_mean,
        score_std=score_std,
        score_min=_safe_min(scores),
        score_max=_safe_max(scores),
        threshold_mean=_mean(thresholds),
        cosine_mean=_mean(cosine),
        dtw_mean=_mean(dtw),
        angle_error_mean=_mean(angle_err),
        mean_pose_distance_mean=_mean(pose_dist),
        top_reference_angle=top_angle,
        top_reference_angle_count=top_angle_count,
        top_feedback_1=top_fb1,
        top_feedback_1_count=top_fb1_count,
        top_feedback_2=top_fb2,
        top_feedback_2_count=top_fb2_count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze existing run metrics.csv files and export per-run summary stats."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="data/runs",
        help="Root directory containing run_*/metrics.csv folders (default: data/runs)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional run folder name, e.g. run_20260323_130223",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/runs/run_metrics_summary.csv",
        help="Summary CSV output path (default: data/runs/run_metrics_summary.csv)",
    )
    parser.add_argument(
        "--print-top",
        type=int,
        default=10,
        help="How many runs to print in console table (default: 10)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    runs_root = _resolve_project_path(args.runs_root)
    if not runs_root.exists():
        print(f"error: runs root not found: {runs_root}")
        return 2

    if args.run_id:
        candidate = runs_root / args.run_id / "metrics.csv"
        metrics_files = [candidate] if candidate.exists() else []
    else:
        metrics_files = sorted(runs_root.glob("run_*/metrics.csv"))

    if not metrics_files:
        print(f"no metrics.csv files found under: {runs_root}")
        return 1

    summaries: list[RunSummary] = []
    for metrics_path in metrics_files:
        try:
            s = summarize_metrics_csv(metrics_path)
        except Exception as exc:
            print(f"warning: failed to parse {metrics_path}: {exc}")
            continue
        if s is not None:
            summaries.append(s)

    if not summaries:
        print("no valid metrics files to summarize")
        return 1

    summaries.sort(key=lambda s: s.run_id)

    output_csv = _resolve_project_path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_id",
        "metrics_path",
        "source",
        "rows",
        "correct",
        "accuracy_pct",
        "score_mean",
        "score_std",
        "score_min",
        "score_max",
        "threshold_mean",
        "cosine_mean",
        "dtw_mean",
        "angle_error_mean",
        "mean_pose_distance_mean",
        "top_reference_angle",
        "top_reference_angle_count",
        "top_feedback_1",
        "top_feedback_1_count",
        "top_feedback_2",
        "top_feedback_2_count",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow(s.__dict__)

    print(f"summarized {len(summaries)} run(s)")
    print(f"written: {output_csv}")

    print("\nTop runs by accuracy:")
    top = sorted(summaries, key=lambda s: s.accuracy_pct, reverse=True)[: max(args.print_top, 1)]
    for idx, s in enumerate(top, start=1):
        print(
            f"{idx:2d}. {s.run_id} | acc={s.accuracy_pct:.2f}% | "
            f"rows={s.rows} | mean={s.score_mean:.2f} | top_angle={s.top_reference_angle}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
