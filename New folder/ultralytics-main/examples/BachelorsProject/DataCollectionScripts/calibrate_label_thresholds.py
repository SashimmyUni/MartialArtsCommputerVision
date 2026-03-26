from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from action_recognition import _best_reference_match, _normalize_key, load_reference_pose_library


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _resolve_project_path(path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate per-label trainer thresholds from reference poses and validation clips. "
            "Expected validation layout: <evaluate_dir>/<technique>/{correct,incorrect}/*.npy"
        )
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="reference_poses",
        help="reference pose root directory",
    )
    parser.add_argument(
        "--evaluate-dir",
        type=str,
        default="datasets/validation",
        help="validation dataset root directory",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="reference_poses/label_thresholds.json",
        help="output JSON file for calibrated thresholds",
    )
    parser.add_argument(
        "--techniques",
        nargs="+",
        default=None,
        help="optional list of techniques to calibrate (defaults to all available)",
    )
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=70.0,
        help="fallback threshold for labels without enough validation data",
    )
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=1,
        help="minimum count required in each class (correct/incorrect) to calibrate",
    )
    return parser.parse_args()


def collect_scores(
    evaluate_root: Path,
    technique_key: str,
    reference_bank: dict[str, np.ndarray],
) -> tuple[list[float], list[float]]:
    base = evaluate_root / technique_key
    correct_files = sorted((base / "correct").glob("*.npy")) if (base / "correct").exists() else []
    incorrect_files = sorted((base / "incorrect").glob("*.npy")) if (base / "incorrect").exists() else []

    correct_scores: list[float] = []
    incorrect_scores: list[float] = []

    for fp in correct_files:
        try:
            seq = np.load(fp)
        except Exception:
            continue
        best = _best_reference_match(seq, reference_bank, technique_key)
        if best is not None:
            correct_scores.append(float(best[1]["score"]))

    for fp in incorrect_files:
        try:
            seq = np.load(fp)
        except Exception:
            continue
        best = _best_reference_match(seq, reference_bank, technique_key)
        if best is not None:
            incorrect_scores.append(float(best[1]["score"]))

    return correct_scores, incorrect_scores


def compute_stats(correct_scores: list[float], incorrect_scores: list[float], threshold: float) -> dict[str, float | int]:
    tp = sum(1 for s in correct_scores if s >= threshold)
    fn = len(correct_scores) - tp
    fp = sum(1 for s in incorrect_scores if s >= threshold)
    tn = len(incorrect_scores) - fp

    total = max(len(correct_scores) + len(incorrect_scores), 1)
    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2.0 * precision * recall / max(precision + recall, 1e-12)) if (precision + recall) > 0 else 0.0
    tpr = recall
    tnr = tn / max(tn + fp, 1)
    balanced_accuracy = 0.5 * (tpr + tnr)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "n_correct": int(len(correct_scores)),
        "n_incorrect": int(len(incorrect_scores)),
    }


def best_threshold(correct_scores: list[float], incorrect_scores: list[float], default_threshold: float) -> dict[str, float | int]:
    if not correct_scores or not incorrect_scores:
        return compute_stats(correct_scores, incorrect_scores, default_threshold)

    candidates = sorted({0.0, 100.0, *correct_scores, *incorrect_scores})
    best: dict[str, float | int] | None = None

    for thr in candidates:
        stats = compute_stats(correct_scores, incorrect_scores, thr)
        if best is None:
            best = stats
            continue

        key = (stats["f1"], stats["balanced_accuracy"], stats["precision"], stats["accuracy"])
        best_key = (best["f1"], best["balanced_accuracy"], best["precision"], best["accuracy"])
        if key > best_key:
            best = stats

    assert best is not None
    return best


def main() -> int:
    args = parse_args()

    reference_dir = _resolve_project_path(args.reference_dir)
    evaluate_dir = _resolve_project_path(args.evaluate_dir)
    output_json = _resolve_project_path(args.output_json)

    references = load_reference_pose_library(str(reference_dir))
    if not references:
        print(f"no references found in {reference_dir}")
        return 2

    if args.techniques:
        techniques = [_normalize_key(t) for t in args.techniques]
    else:
        techniques = sorted(references.keys())

    thresholds: dict[str, float] = {}
    metrics: dict[str, dict[str, float | int | str]] = {}

    for technique in techniques:
        if technique not in references:
            continue

        ref_bank = references[technique]
        correct_scores, incorrect_scores = collect_scores(evaluate_dir, technique, ref_bank)

        if len(correct_scores) < args.min_samples_per_class or len(incorrect_scores) < args.min_samples_per_class:
            thresholds[technique] = float(args.default_threshold)
            metrics[technique] = {
                "status": "fallback_default",
                "reason": "insufficient_samples",
                "n_correct": len(correct_scores),
                "n_incorrect": len(incorrect_scores),
                "threshold": float(args.default_threshold),
            }
            print(
                f"{technique}: insufficient samples (correct={len(correct_scores)}, incorrect={len(incorrect_scores)}), "
                f"using default {args.default_threshold:.1f}"
            )
            continue

        best = best_threshold(correct_scores, incorrect_scores, float(args.default_threshold))
        thr = float(best["threshold"])
        thresholds[technique] = thr
        metrics[technique] = {"status": "calibrated", **best}
        print(
            f"{technique}: threshold={thr:.2f} f1={float(best['f1']):.3f} "
            f"bal_acc={float(best['balanced_accuracy']):.3f} "
            f"(n+={len(correct_scores)}, n-={len(incorrect_scores)})"
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_dir": str(reference_dir),
        "evaluate_dir": str(evaluate_dir),
        "default_threshold": float(args.default_threshold),
        "objective": "maximize_f1_then_balanced_accuracy",
        "thresholds": thresholds,
        "metrics": metrics,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"written: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
