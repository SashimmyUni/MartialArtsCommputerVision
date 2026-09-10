"""Continuous multi-fighter technique identification for fight footage.

Watches up to ``--num-fighters`` tracked people (default 2) and, for each,
fuses a zero-shot video-classifier proposal with DTW reference-match
confirmation into one technique label per classifier tick, then collapses
consecutive ticks into discrete technique **events** for usage-statistics
reporting (see ``scripts/analyze_fight_metrics.py``).

This is deliberately a separate entry point from ``action_recognition.py``,
not a mode grafted onto ``run()``: that function's scoring/metrics/trainer
state is hardwired to one tracked person scored against one
``--target-technique``, and reworking it in place would risk its pinned
equivalence tests for no benefit to the single-person coaching trainer. This
script instead imports the scoring/classifier primitives it needs
(``load_reference_pose_library``, ``_best_reference_match``,
``HuggingFaceVideoClassifier``, pose normalization, ``_track_activity_score``)
and drives its own loop.

Scope for this phase: standing striking only (punches, elbows, knees, kicks).
Takedowns, clinch control, ground position, ground-and-pound and submissions
are not modeled. Fighter identity is a best-effort appearance heuristic (see
``fighter_identity.py``), not a trained person-ReID model.

Usage:
    python fight_analysis.py --source <video_or_youtube_url> --no-display
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.data.loaders import get_best_youtube_url
from ultralytics.utils.plotting import Annotator
from ultralytics.utils.tqdm import TQDM
from ultralytics.utils.torch_utils import select_device

import action_recognition as ar
from fighter_identity import FighterIdentityRegistry, torso_histogram
from technique_catalog import MMA_CATALOG_PATH, resolve_reference_key, technique_family
from technique_events import FramePrediction, segment_events

PROJECT_ROOT = Path(__file__).resolve().parent

#: Fixed colors per fighter slot so the same fighter reads consistently frame
#: to frame; anything past this (gallery overflow) falls back to white.
_FIGHTER_COLORS = [(0, 255, 0), (0, 165, 255), (255, 255, 0), (255, 0, 255)]


def _fighter_color(fighter_id: str) -> tuple[int, int, int]:
    try:
        idx = int(fighter_id.rsplit("_", 1)[-1]) - 1
    except ValueError:
        idx = -1
    if 0 <= idx < len(_FIGHTER_COLORS):
        return _FIGHTER_COLORS[idx]
    return (255, 255, 255)


def _select_active_tracks(
    track_ids: list[int],
    track_boxes: dict[int, np.ndarray],
    track_kpts_history: dict[int, list[np.ndarray]],
    track_box_history: dict[int, list[np.ndarray]],
    frame_width: int,
    frame_height: int,
    top_n: int,
) -> tuple[list[int], dict[int, float], dict[int, np.ndarray]]:
    """Rank all currently-tracked people by the shared activity-score
    heuristic (``action_recognition._track_activity_score``) and return the
    top ``top_n`` -- e.g. the two fighters, filtering out a referee or
    cornerman who wanders into frame. Generalizes
    ``action_recognition._select_primary_track`` from one person to N.
    """
    if not track_ids:
        return [], {}, {}

    scores: dict[int, float] = {}
    stacked_by_track: dict[int, np.ndarray] = {}
    for track_id in track_ids:
        score, stacked = ar._track_activity_score(
            kpt_history=track_kpts_history.get(track_id, []),
            box_history=track_box_history.get(track_id, []),
            current_box=track_boxes[track_id],
            frame_width=frame_width,
            frame_height=frame_height,
        )
        scores[track_id] = score
        if stacked is not None:
            stacked_by_track[track_id] = stacked

    ranked = sorted(scores, key=scores.get, reverse=True)
    return ranked[:top_n], scores, stacked_by_track


def _fuse_prediction(
    frame_counter: int,
    fps: float,
    candidate_labels: list[str],
    candidate_confs: list[float],
    kpts_window: np.ndarray | None,
    references: dict[str, dict[str, np.ndarray]],
    min_confidence: float,
    dtw_confirm_threshold: float,
) -> FramePrediction | None:
    """Fuse a zero-shot classifier's ranked candidates with DTW confirmation.

    Tries each candidate label in confidence order and takes the first one
    that both has a captured reference bank and clears
    ``dtw_confirm_threshold`` (``detection_method="hybrid"``). Falls back to
    the top-1 classifier label alone when no candidate confirms -- e.g. no
    reference bank exists yet for that technique -- as long as its confidence
    clears ``min_confidence`` (``detection_method="classifier_only"``).
    Returns ``None`` for a tick with no sufficiently confident guess.
    """
    if not candidate_labels or candidate_confs[0] < min_confidence:
        return None

    time_s = frame_counter / fps if fps > 0 else float(frame_counter)
    top_label = candidate_labels[0]
    top_conf = float(candidate_confs[0])
    top_key = ar._normalize_key(top_label)

    if kpts_window is not None:
        pose_norm = ar.normalize_pose_sequence(kpts_window, conf_thresh=0.2)
        for label, conf in zip(candidate_labels, candidate_confs):
            technique_key = ar._normalize_key(label)
            bank_key = resolve_reference_key(references, technique_key, path=MMA_CATALOG_PATH)
            if bank_key is None:
                continue
            best = ar._best_reference_match(
                user_sequence=kpts_window,
                reference_bank=references[bank_key],
                technique=technique_key,
                user_norm=pose_norm,
            )
            if best is None:
                continue
            _, metrics = best
            score = float(metrics["score"])
            if score >= dtw_confirm_threshold:
                return FramePrediction(
                    frame=frame_counter,
                    time_s=time_s,
                    technique=technique_key,
                    family=technique_family(technique_key, path=MMA_CATALOG_PATH),
                    confidence=float(conf),
                    detection_method="hybrid",
                    dtw_score=score,
                )

    return FramePrediction(
        frame=frame_counter,
        time_s=time_s,
        technique=top_key,
        family=technique_family(top_key, path=MMA_CATALOG_PATH),
        confidence=top_conf,
        detection_method="classifier_only",
        dtw_score=None,
    )


def _open_capture(source: str) -> tuple[cv2.VideoCapture, str]:
    """Open ``source`` (webcam index, YouTube URL, or video file). Mirrors
    the source-resolution logic in ``action_recognition.run()``.
    """
    if source.isdigit():
        return cv2.VideoCapture(int(source)), source
    if source.startswith("http") and urlparse(source).hostname in {"www.youtube.com", "youtube.com", "youtu.be"}:
        resolved = source
        try:
            resolved = get_best_youtube_url(source)
        except Exception as e:
            print(f"warning: failed to select best YouTube stream ({e}), using original URL")
        return cv2.VideoCapture(resolved), source
    if Path(source).suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
        raise ValueError(
            f"Invalid source '{source}'. Supported sources are: webcam index (e.g. 0), "
            "YouTube URLs, or video files (.mp4, .avi, .mov, .mkv, .webm, .m4v)."
        )
    return cv2.VideoCapture(source), source


def _write_events_csv(path: Path, events: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fighter_id",
        "technique",
        "family",
        "detection_method",
        "start_frame",
        "end_frame",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "confidence",
        "dtw_score",
        "frame_count",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ev, source in events:
            writer.writerow(
                {
                    "fighter_id": ev.fighter_id,
                    "technique": ev.technique,
                    "family": ev.family or "",
                    "detection_method": ev.detection_method,
                    "start_frame": ev.start_frame,
                    "end_frame": ev.end_frame,
                    "start_time_s": f"{ev.start_time_s:.3f}",
                    "end_time_s": f"{ev.end_time_s:.3f}",
                    "duration_s": f"{ev.duration_s:.3f}",
                    "confidence": f"{ev.confidence:.4f}",
                    "dtw_score": f"{ev.dtw_score:.4f}" if ev.dtw_score is not None else "",
                    "frame_count": ev.frame_count,
                    "source": source,
                }
            )


def _print_summary(events_by_fighter: dict[str, list]) -> None:
    for fighter_id in sorted(events_by_fighter):
        events = events_by_fighter[fighter_id]
        counts = Counter(ev.technique for ev in events)
        print(f"{fighter_id}: {len(events)} technique event(s)")
        for technique, count in counts.most_common():
            print(f"  {technique:20s} x{count}")


def run_fight_analysis(
    weights: str = "yolo26n-pose.pt",
    device: str = "",
    source: str = "0",
    reference_dir: str = "reference_poses",
    num_fighters: int = 2,
    crop_margin_percentage: int = 10,
    num_video_sequence_samples: int = 8,
    skip_frame: int = 1,
    video_cls_overlap_ratio: float = 0.25,
    video_classifier_model: str = "microsoft/xclip-base-patch32",
    label_set: str = "mma",
    labels: list[str] | None = None,
    classifier_topk: int = 3,
    classifier_min_confidence: float = 0.15,
    dtw_confirm_threshold: float = 45.0,
    min_event_duration_ticks: int = 1,
    max_event_gap_ticks: int = 1,
    identity_match_threshold: float = 0.55,
    identity_momentum: float = 0.8,
    fp16: bool = False,
    imgsz: int = 640,
    detect_stride: int = 1,
    storage_root: str = "data",
    run_name: str | None = None,
    enable_structured_storage: bool = True,
    events_csv: str | None = None,
    output_path: str | None = None,
    draw_boxes: bool = True,
    display: bool = True,
    debug: bool = False,
) -> dict[str, list]:
    """Run continuous multi-fighter technique identification on a video source.

    Returns the collapsed technique events, keyed by fighter_id, so callers
    embedding this (tests, notebooks) don't have to re-read the CSV.
    """
    if num_fighters < 1:
        raise ValueError("--num-fighters must be >= 1")
    if skip_frame <= 0:
        raise ValueError("--skip-frame must be >= 1")
    if num_video_sequence_samples <= 0:
        raise ValueError("--num-video-sequence-samples must be >= 1")
    if not (0.0 <= video_cls_overlap_ratio < 1.0):
        raise ValueError("--video-cls-overlap-ratio must be in [0.0, 1.0)")
    if classifier_topk < 1:
        raise ValueError("--classifier-topk must be >= 1")
    if imgsz <= 0:
        raise ValueError("--imgsz must be >= 1")
    if detect_stride <= 0:
        raise ValueError("--detect-stride must be >= 1")
    if video_classifier_model in ar.TorchVisionVideoClassifier.available_model_names():
        raise ValueError(
            "fight_analysis.py requires a zero-shot classifier with custom labels "
            "(e.g. an xclip model) -- TorchVisionVideoClassifier only predicts fixed "
            "Kinetics-400 labels and can't be fused with the MMA technique catalogue."
        )

    if isinstance(output_path, str) and output_path.strip().lower() in {"", "none", "null", "off"}:
        output_path = None

    video_cls_step = max(1, int(round(num_video_sequence_samples * skip_frame * (1.0 - video_cls_overlap_ratio))))
    max_gap_frames = video_cls_step * max(1, max_event_gap_ticks)

    if not labels:
        labels = ar.LABEL_SETS[label_set].copy()

    weights = str(ar._resolve_project_path(weights))
    reference_dir = str(ar._resolve_project_path(reference_dir))
    storage_root = str(ar._resolve_project_path(storage_root))
    if output_path:
        output_path = str(ar._resolve_project_path(output_path))
    if events_csv:
        events_csv = str(ar._resolve_project_path(events_csv))

    source_path = Path(source)
    if not source.isdigit() and not source.startswith("http") and source_path.suffix:
        source = str(ar._resolve_project_path(source))

    run_config = {
        "weights": weights,
        "device": device,
        "source": source,
        "reference_dir": reference_dir,
        "num_fighters": num_fighters,
        "num_video_sequence_samples": num_video_sequence_samples,
        "skip_frame": skip_frame,
        "video_cls_overlap_ratio": video_cls_overlap_ratio,
        "video_classifier_model": video_classifier_model,
        "labels": labels,
        "classifier_topk": classifier_topk,
        "classifier_min_confidence": classifier_min_confidence,
        "dtw_confirm_threshold": dtw_confirm_threshold,
        "min_event_duration_ticks": min_event_duration_ticks,
        "max_event_gap_ticks": max_event_gap_ticks,
        "identity_match_threshold": identity_match_threshold,
        "identity_momentum": identity_momentum,
        "imgsz": imgsz,
        "detect_stride": detect_stride,
    }

    storage_ctx = None
    if enable_structured_storage:
        storage_ctx = ar._init_structured_run_storage(storage_root=storage_root, run_name=run_name, run_config=run_config)
        print(f"structured storage run_id: {storage_ctx['run_id']}")
        print(f"structured storage dir: {storage_ctx['run_dir']}")
        if events_csv is None:
            events_csv = str(Path(storage_ctx["run_dir"]) / "technique_events.csv")
    elif events_csv is None:
        events_csv = str(PROJECT_ROOT / "technique_events.csv")

    device = select_device(device)
    yolo_model = YOLO(weights).to(device)
    is_pose_model = getattr(yolo_model, "task", "") == "pose"
    if not is_pose_model:
        raise ValueError("fight_analysis.py requires a pose model (e.g. yolo26n-pose.pt) for DTW confirmation.")
    pose_half = bool(fp16) and torch.device(device).type == "cuda"

    references = ar.load_reference_pose_library(reference_dir)
    print(f"loaded reference techniques: {sorted(references.keys())}")
    video_classifier = ar.HuggingFaceVideoClassifier(labels, model_name=video_classifier_model, device=device, fp16=fp16)
    identity_registry = FighterIdentityRegistry(
        max_fighters=num_fighters, match_threshold=identity_match_threshold, embedding_momentum=identity_momentum
    )

    cap, source = _open_capture(source)
    if not cap.isOpened():
        raise ValueError(f"Failed to open source: {source}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = total_frames_raw if total_frames_raw > 0 else None

    out = None
    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            raise ValueError(f"Failed to open output writer: {out_path}")

    track_history: dict[int, list[np.ndarray]] = defaultdict(list)
    track_kpts_history: dict[int, list[np.ndarray]] = defaultdict(list)
    track_box_history: dict[int, list[np.ndarray]] = defaultdict(list)
    track_to_fighter: dict[int, str] = {}
    fighter_state: dict[str, dict[str, object]] = {}
    predictions_by_fighter: dict[str, list[FramePrediction]] = defaultdict(list)

    frame_counter = 0
    cached_results = None
    frame_progress = TQDM(total=total_frames, desc="processing frames", unit="frame")

    loop_start = time.perf_counter()
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_counter += 1

            if detect_stride <= 1 or cached_results is None or (frame_counter - 1) % detect_stride == 0:
                results = yolo_model.track(frame, persist=True, classes=[0], verbose=False, imgsz=imgsz, half=pose_half)
                cached_results = results
            else:
                results = cached_results

            annotator = Annotator(frame, line_width=3, font_size=10, pil=False)

            if not results[0].boxes.is_track:
                if display:
                    cv2.imshow("Fight Analysis", annotator.im)
                if out is not None:
                    out.write(annotator.im)
                frame_progress.update(1)
                if display and (cv2.waitKey(1) & 0xFF == ord("q")):
                    break
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            pose_instances = ar.extract_pose_instances(results[0], conf_thres=0.01)

            current_track_ids: list[int] = []
            current_track_boxes: dict[int, np.ndarray] = {}
            track_to_pose: dict[int, np.ndarray] = {}
            for i, (box, track_id_raw) in enumerate(zip(boxes, track_ids)):
                track_id = int(track_id_raw)
                current_track_ids.append(track_id)
                current_track_boxes[track_id] = np.asarray(box, dtype=np.float32)
                if pose_instances and i < len(pose_instances):
                    track_to_pose[track_id] = pose_instances[i]

                if frame_counter % skip_frame == 0:
                    crop = ar.crop_and_pad(frame, box, crop_margin_percentage)
                    track_history[track_id].append(crop)
                    track_box_history[track_id].append(np.asarray(box, dtype=np.float32))
                    if pose_instances and i < len(pose_instances):
                        sanitized = ar._sanitize_kpt_entry(pose_instances[i])
                        if sanitized is not None:
                            track_kpts_history[track_id].append(sanitized)

                if len(track_history[track_id]) > num_video_sequence_samples:
                    track_history[track_id].pop(0)
                if len(track_box_history.get(track_id, [])) > num_video_sequence_samples:
                    track_box_history[track_id].pop(0)
                if len(track_kpts_history.get(track_id, [])) > num_video_sequence_samples:
                    track_kpts_history[track_id].pop(0)

            active_track_ids, track_scores, stacked_kpts_by_track = _select_active_tracks(
                track_ids=current_track_ids,
                track_boxes=current_track_boxes,
                track_kpts_history=track_kpts_history,
                track_box_history=track_box_history,
                frame_width=frame_width,
                frame_height=frame_height,
                top_n=num_fighters,
            )
            if debug and active_track_ids:
                scores_text = ", ".join(f"{tid}={track_scores.get(tid, 0.0):.2f}" for tid in active_track_ids)
                print(f"[debug] frame {frame_counter}: active tracks {scores_text}")

            for track_id in active_track_ids:
                if frame_counter % skip_frame == 0 and track_history.get(track_id):
                    embedding = torso_histogram(track_history[track_id][-1])
                    fighter_id = identity_registry.resolve(track_id, embedding)
                    track_to_fighter[track_id] = fighter_id

            if (
                frame_counter % video_cls_step == 0
                and active_track_ids
            ):
                tick_track_ids = [
                    tid for tid in active_track_ids if len(track_history.get(tid, [])) >= num_video_sequence_samples
                ]
                if tick_track_ids:
                    crops_batch = torch.cat(
                        [
                            video_classifier.preprocess_crops_for_video_cls(track_history[tid][-num_video_sequence_samples:])
                            for tid in tick_track_ids
                        ],
                        dim=0,
                    )
                    logits = video_classifier(crops_batch)
                    probs = logits.softmax(dim=-1)
                    k = min(classifier_topk, probs.shape[-1])
                    topk_vals, topk_idx = probs.topk(k, dim=-1)

                    for row, track_id in enumerate(tick_track_ids):
                        fighter_id = track_to_fighter.get(track_id)
                        if fighter_id is None:
                            continue
                        candidate_labels = [video_classifier.labels[i] for i in topk_idx[row].tolist()]
                        candidate_confs = [float(v) for v in topk_vals[row].tolist()]

                        kpts_window = None
                        stacked = stacked_kpts_by_track.get(track_id)
                        if stacked is None:
                            stacked = ar._safe_stack_kpt_sequence(track_kpts_history.get(track_id, []))
                        if stacked is not None and len(stacked) >= num_video_sequence_samples:
                            kpts_window = stacked[-num_video_sequence_samples:]

                        prediction = _fuse_prediction(
                            frame_counter=frame_counter,
                            fps=fps,
                            candidate_labels=candidate_labels,
                            candidate_confs=candidate_confs,
                            kpts_window=kpts_window,
                            references=references,
                            min_confidence=classifier_min_confidence,
                            dtw_confirm_threshold=dtw_confirm_threshold,
                        )
                        if prediction is not None:
                            predictions_by_fighter[fighter_id].append(prediction)
                            fighter_state[fighter_id] = {
                                "technique": prediction.technique,
                                "confidence": prediction.confidence,
                                "detection_method": prediction.detection_method,
                            }
                            if debug:
                                print(
                                    f"[debug] frame {frame_counter}: {fighter_id} candidates={list(zip(candidate_labels, [round(c, 2) for c in candidate_confs]))} "
                                    f"-> {prediction.technique} ({prediction.detection_method}, dtw={prediction.dtw_score})"
                                )

            if draw_boxes:
                for track_id in active_track_ids:
                    box = current_track_boxes.get(track_id)
                    fighter_id = track_to_fighter.get(track_id)
                    if box is None or fighter_id is None:
                        continue
                    state = fighter_state.get(fighter_id)
                    if state:
                        label_text = (
                            f"{fighter_id} | {state['technique']} {float(state['confidence']):.2f} "
                            f"({state['detection_method']})"
                        )
                    else:
                        label_text = fighter_id
                    annotator.box_label(box, label_text, color=_fighter_color(fighter_id))

            display_frame = annotator.im
            if display:
                cv2.imshow("Fight Analysis", display_frame)
            if out is not None:
                out.write(display_frame)
            frame_progress.update(1)
            if display and (cv2.waitKey(1) & 0xFF == ord("q")):
                break
    finally:
        events_by_fighter: dict[str, list] = {}
        for fighter_id, predictions in predictions_by_fighter.items():
            events_by_fighter[fighter_id] = segment_events(
                predictions,
                fighter_id=fighter_id,
                min_duration_ticks=min_event_duration_ticks,
                max_gap_frames=max_gap_frames,
            )

        if events_csv:
            all_events = [(ev, source) for events in events_by_fighter.values() for ev in events]
            all_events.sort(key=lambda item: item[0].start_frame)
            _write_events_csv(Path(events_csv), all_events)
            print(f"wrote {len(all_events)} technique event(s) to: {events_csv}")

        _print_summary(events_by_fighter)

        loop_elapsed = time.perf_counter() - loop_start
        print(f"timing: {frame_counter} frames in {loop_elapsed:.2f}s ({frame_counter / max(loop_elapsed, 1e-6):.1f} fps)")

        cap.release()
        if out is not None:
            out.release()
        frame_progress.close()
        cv2.destroyAllWindows()

    return events_by_fighter


def parse_opt() -> argparse.Namespace:
    """Parse command line arguments for the fight-analysis pipeline."""

    def _positive_int(value: str) -> int:
        ivalue = int(value)
        if ivalue < 1:
            raise argparse.ArgumentTypeError("value must be >= 1")
        return ivalue

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=str, default="yolo26n-pose.pt", help="YOLO pose weights")
    parser.add_argument("--device", default="", help='cuda device, e.g. 0 or cpu/mps, "" for auto-detection')
    parser.add_argument("--source", type=str, default="0", help="video file path, youtube URL, or webcam index")
    parser.add_argument("--reference-dir", type=str, default="reference_poses", help="reference pose root")
    parser.add_argument("--num-fighters", type=_positive_int, default=2, help="number of fighters to track (default: 2)")
    parser.add_argument("--crop-margin-percentage", type=int, default=10)
    parser.add_argument("--num-video-sequence-samples", type=_positive_int, default=8)
    parser.add_argument("--skip-frame", type=_positive_int, default=1)
    parser.add_argument("--video-cls-overlap-ratio", type=float, default=0.25)
    parser.add_argument("--video-classifier-model", type=str, default="microsoft/xclip-base-patch32")
    parser.add_argument(
        "--label-set",
        type=str,
        choices=sorted(ar.LABEL_SETS),
        default="mma",
        help="named zero-shot label set to use when --labels is not given (default: mma)",
    )
    parser.add_argument("--labels", nargs="+", type=str, default=None, help="override --label-set with a custom label list")
    parser.add_argument(
        "--classifier-topk",
        type=_positive_int,
        default=3,
        help="how many ranked classifier candidates to attempt DTW confirmation on (default: 3)",
    )
    parser.add_argument(
        "--classifier-min-confidence",
        type=float,
        default=0.15,
        help="drop a tick's prediction entirely below this top-1 softmax confidence (default: 0.15)",
    )
    parser.add_argument(
        "--dtw-confirm-threshold",
        type=float,
        default=45.0,
        help=(
            "minimum _best_reference_match score (0-100) for a classifier candidate to count as "
            "DTW-confirmed ('hybrid'); lower than the coaching trainer's 70.0 default because this "
            "is identification against fast, in-fight footage, not a quality gate (default: 45.0)"
        ),
    )
    parser.add_argument("--min-event-duration-ticks", type=_positive_int, default=1)
    parser.add_argument("--max-event-gap-ticks", type=_positive_int, default=1)
    parser.add_argument("--identity-match-threshold", type=float, default=0.55)
    parser.add_argument("--identity-momentum", type=float, default=0.8)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--imgsz", type=_positive_int, default=640)
    parser.add_argument("--detect-stride", type=_positive_int, default=1)
    parser.add_argument("--storage-root", type=str, default="data")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--disable-structured-storage", dest="enable_structured_storage", action="store_false"
    )
    parser.add_argument("--events-csv", type=str, default=None, help="override where technique_events.csv is written")
    parser.add_argument("--output-path", type=str, default=None, help="optional annotated output video path")
    parser.add_argument("--no-boxes", dest="draw_boxes", action="store_false")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main(opt: argparse.Namespace) -> None:
    if getattr(opt, "run_name", None) in (None, ""):
        source = getattr(opt, "source", None)
        if source and not source.isdigit() and not source.startswith("http"):
            opt.run_name = Path(source).stem
    options = vars(opt).copy()
    label_set = options.pop("label_set", "mma")
    if not options.get("labels"):
        options["labels"] = ar.LABEL_SETS[label_set].copy()
    run_fight_analysis(**options)


if __name__ == "__main__":
    main(parse_opt())
