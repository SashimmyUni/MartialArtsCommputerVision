# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
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


MARTIAL_ARTS_LABELS = [
    "fighting stance",
    "jab",
    "cross",
    "hook",
    "uppercut",
    "front kick",
    "roundhouse kick",
    "side kick",
    "back kick",
    "spinning back kick",
    "knee strike",
    "elbow strike",
    "axe kick",
]

PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_project_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    p = Path(path_value)
    return p if p.is_absolute() else PROJECT_ROOT / p


class TorchVisionVideoClassifier:
    """Video classifier using pretrained TorchVision models for action recognition.

    This class provides an interface for video classification using various pretrained models from TorchVision's video
    model collection, supporting models like S3D, R3D, Swin3D, and MViT architectures.

    Attributes:
        model (torch.nn.Module): The loaded TorchVision model for video classification.
        weights (torchvision.models.video.Weights): The weights used for the model.
        device (torch.device): The device on which the model is loaded.

    Methods:
        available_model_names: Returns a list of available model names.
        preprocess_crops_for_video_cls: Preprocesses crops for video classification.
        __call__: Performs inference on the given sequences.
        postprocess: Postprocesses the model's output.

    Examples:
        >>> classifier = TorchVisionVideoClassifier("s3d", device="cpu")
        >>> crops = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(8)]
        >>> tensor = classifier.preprocess_crops_for_video_cls(crops)
        >>> outputs = classifier(tensor)
        >>> labels, confidences = classifier.postprocess(outputs)

    References:
        https://pytorch.org/vision/stable/
    """

    from torchvision.models.video import (
        MViT_V1_B_Weights,
        MViT_V2_S_Weights,
        R3D_18_Weights,
        S3D_Weights,
        Swin3D_B_Weights,
        Swin3D_T_Weights,
        mvit_v1_b,
        mvit_v2_s,
        r3d_18,
        s3d,
        swin3d_b,
        swin3d_t,
    )

    model_name_to_model_and_weights = {
        "s3d": (s3d, S3D_Weights.DEFAULT),
        "r3d_18": (r3d_18, R3D_18_Weights.DEFAULT),
        "swin3d_t": (swin3d_t, Swin3D_T_Weights.DEFAULT),
        "swin3d_b": (swin3d_b, Swin3D_B_Weights.DEFAULT),
        "mvit_v1_b": (mvit_v1_b, MViT_V1_B_Weights.DEFAULT),
        "mvit_v2_s": (mvit_v2_s, MViT_V2_S_Weights.DEFAULT),
    }

    def __init__(self, model_name: str, device: str | torch.device = ""):
        """Initialize the VideoClassifier with the specified model name and device.

        Args:
            model_name (str): The name of the model to use. Must be one of the available models.
            device (str | torch.device): The device to run the model on.
        """
        if model_name not in self.model_name_to_model_and_weights:
            raise ValueError(f"Invalid model name '{model_name}'. Available models: {self.available_model_names()}")
        model, self.weights = self.model_name_to_model_and_weights[model_name]
        self.device = select_device(device)
        self.model = model(weights=self.weights).to(self.device).eval()

    @staticmethod
    def available_model_names() -> list[str]:
        """Get the list of available model names.

        Returns:
            (list[str]): List of available model names that can be used with this classifier.
        """
        return list(TorchVisionVideoClassifier.model_name_to_model_and_weights.keys())

    def preprocess_crops_for_video_cls(
        self, crops: list[np.ndarray], input_size: list[int] | None = None
    ) -> torch.Tensor:
        """Preprocess a list of crops for video classification.

        Args:
            crops (list[np.ndarray]): List of crops to preprocess. Each crop should have dimensions (H, W, C).
            input_size (list[int], optional): The target input size for the model.

        Returns:
            (torch.Tensor): Preprocessed crops as a tensor with dimensions (1, C, T, H, W).
        """
        if input_size is None:
            input_size = [224, 224]
        from torchvision.transforms import v2

        transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(input_size, antialias=True),
                v2.Normalize(mean=self.weights.transforms().mean, std=self.weights.transforms().std),
            ]
        )

        processed_crops = [transform(torch.from_numpy(crop).permute(2, 0, 1)) for crop in crops]
        return torch.stack(processed_crops).unsqueeze(0).permute(0, 2, 1, 3, 4).to(self.device)

    def __call__(self, sequences: torch.Tensor) -> torch.Tensor:
        """Perform inference on the given sequences.

        Args:
            sequences (torch.Tensor): The input sequences for the model with dimensions (B, C, T, H, W) for batched
                video frames or (C, T, H, W) for single video frames.

        Returns:
            (torch.Tensor): The model's output logits.
        """
        with torch.inference_mode():
            return self.model(sequences)

    def postprocess(self, outputs: torch.Tensor) -> tuple[list[str], list[float]]:
        """Postprocess the model's batch output.

        Args:
            outputs (torch.Tensor): The model's output logits.

        Returns:
            (tuple[list[str], list[float]]): Predicted labels and their confidence scores.
        """
        pred_labels = []
        pred_confs = []
        for output in outputs:
            pred_class = output.argmax(0).item()
            pred_label = self.weights.meta["categories"][pred_class]
            pred_labels.append(pred_label)
            pred_conf = output.softmax(0)[pred_class].item()
            pred_confs.append(pred_conf)

        return pred_labels, pred_confs


class HuggingFaceVideoClassifier:
    """Zero-shot video classifier using Hugging Face transformer models.

    This class provides an interface for zero-shot video classification using Hugging Face models, supporting custom
    label sets and various transformer architectures for video understanding.

    Attributes:
        fp16 (bool): Whether to use FP16 for inference.
        labels (list[str]): List of labels for zero-shot classification.
        device (torch.device): The device on which the model is loaded.
        processor (transformers.AutoProcessor): The processor for the model.
        model (transformers.AutoModel): The loaded Hugging Face model.

    Methods:
        preprocess_crops_for_video_cls: Preprocesses crops for video classification.
        __call__: Performs inference on the given sequences.
        postprocess: Postprocesses the model's output.

    Examples:
        >>> labels = ["walking", "running", "dancing"]
        >>> classifier = HuggingFaceVideoClassifier(labels, device="cpu")
        >>> crops = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(8)]
        >>> tensor = classifier.preprocess_crops_for_video_cls(crops)
        >>> outputs = classifier(tensor)
        >>> labels, confidences = classifier.postprocess(outputs)
    """

    def __init__(
        self,
        labels: list[str],
        model_name: str = "microsoft/xclip-base-patch16-zero-shot",
        device: str | torch.device = "",
        fp16: bool = False,
    ):
        """Initialize the HuggingFaceVideoClassifier with the specified model name.

        Args:
            labels (list[str]): List of labels for zero-shot classification.
            model_name (str): The name of the model to use.
            device (str | torch.device): The device to run the model on.
            fp16 (bool): Whether to use FP16 for inference.
        """
        self.fp16 = fp16
        self.labels = labels
        self.device = select_device(device)
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Hugging Face classifier requires the 'transformers' package. "
                "Install it with: pip install transformers"
            ) from exc
        self.processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device)
        if fp16:
            model = model.half()
        self.model = model.eval()

    def preprocess_crops_for_video_cls(
        self, crops: list[np.ndarray], input_size: list[int] | None = None
    ) -> torch.Tensor:
        """Preprocess a list of crops for video classification.

        Args:
            crops (list[np.ndarray]): List of crops to preprocess. Each crop should have dimensions (H, W, C).
            input_size (list[int], optional): The target input size for the model.

        Returns:
            (torch.Tensor): Preprocessed crops as a tensor with dimensions (1, T, C, H, W).
        """
        if input_size is None:
            input_size = [224, 224]
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.float() / 255.0),
                transforms.Resize(input_size),
                transforms.Normalize(
                    mean=self.processor.image_processor.image_mean, std=self.processor.image_processor.image_std
                ),
            ]
        )

        processed_crops = [transform(torch.from_numpy(crop).permute(2, 0, 1)) for crop in crops]  # (T, C, H, W)
        output = torch.stack(processed_crops).unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        if self.fp16:
            output = output.half()
        return output

    def __call__(self, sequences: torch.Tensor) -> torch.Tensor:
        """Perform inference on the given sequences.

        Args:
            sequences (torch.Tensor): Batched input video frames with shape (B, T, C, H, W).

        Returns:
            (torch.Tensor): The model's output logits.
        """
        input_ids = self.processor(text=self.labels, return_tensors="pt", padding=True)["input_ids"].to(self.device)

        inputs = {"pixel_values": sequences, "input_ids": input_ids}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        return outputs.logits_per_video

    def postprocess(self, outputs: torch.Tensor) -> tuple[list[list[str]], list[list[float]]]:
        """Postprocess the model's batch output.

        Args:
            outputs (torch.Tensor): The model's output logits.

        Returns:
            (tuple[list[list[str]], list[list[float]]]): Predicted top-2 labels and confidence scores for each sample.
        """
        pred_labels = []
        pred_confs = []

        with torch.no_grad():
            logits_per_video = outputs  # Assuming outputs is already the logits tensor
            probs = logits_per_video.softmax(dim=-1)  # Use softmax to convert logits to probabilities

        for prob in probs:
            k = min(2, len(self.labels))
            topk_indices = prob.topk(k).indices.tolist()
            top2_labels = [self.labels[idx] for idx in topk_indices]
            top2_confs = prob[topk_indices].tolist()
            pred_labels.append(top2_labels)
            pred_confs.append(top2_confs)

        return pred_labels, pred_confs




def save_keypoint_sequences(track_kpts_history: dict[int, list[np.ndarray]], out_dir: str) -> None:
    """Save stored keypoints for each track to disk.

    Each track's sequence is written as a NumPy ``.npy`` file named
    ``track_<id>.npy``.  The output directory is created if necessary.
    """
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    non_empty_tracks = [tid for tid, seq in track_kpts_history.items() if len(seq) > 0]
    print(f"saving keypoints: {len(non_empty_tracks)} track(s) to {path.resolve()}")

    saved_tracks = 0
    for tid, seq in track_kpts_history.items():
        if len(seq) == 0:
            continue

        stacked = _safe_stack_kpt_sequence(seq)
        if stacked is None:
            # Fallback: persist each sanitized frame separately.
            fallback_saved = 0
            for fi, raw in enumerate(seq):
                frame_arr = _sanitize_kpt_entry(raw)
                if frame_arr is None:
                    continue
                frame_file = path / f"track_{tid}_frame_{fi:04d}.npy"
                np.save(frame_file, frame_arr.astype(np.float32))
                print(f"wrote {frame_file}")
                fallback_saved += 1
            if fallback_saved > 0:
                saved_tracks += 1
            continue

        arr = stacked  # shape (T, num_kpts, 2|3)
        out_file = path / f"track_{tid}.npy"
        np.save(out_file, arr)
        print(f"wrote {out_file}")
        saved_tracks += 1

    print(f"saved keypoint data for {saved_tracks} track(s)")


def draw_pose_on_blank(kpts: np.ndarray, shape: tuple[int,int]) -> np.ndarray:
    """Return a blank image with the pose skeleton drawn.

    Args:
        kpts (np.ndarray): Array of keypoints shape (N,2) or (N,3).
        shape (tuple[int,int]): (height, width) of canvas.
    """
    canvas = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    annot = Annotator(canvas, line_width=3, font_size=10, pil=False)
    annot.kpts(kpts, shape=shape)
    return annot.im


COCO17_EDGES = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (5, 6),
    (11, 12),
    (5, 11), (6, 12),
    (0, 1), (0, 2),
    (1, 3), (2, 4),
]


def draw_pose_cv2(
    image: np.ndarray,
    kpts: np.ndarray,
    conf_thres: float = 0.05,
    joint_color: tuple[int, int, int] = (0, 170, 255),
    edge_color: tuple[int, int, int] = (0, 120, 230),
    joint_radius: int = 4,
    edge_thickness: int = 2,
) -> None:
    """Draw keypoints and COCO17 skeleton directly with OpenCV."""
    if kpts.ndim != 2 or kpts.shape[1] < 2:
        return

    n = kpts.shape[0]

    def valid_idx(i: int) -> bool:
        if i >= n:
            return False
        x, y = float(kpts[i, 0]), float(kpts[i, 1])
        if not (np.isfinite(x) and np.isfinite(y)):
            return False
        if kpts.shape[1] >= 3:
            c = float(kpts[i, 2])
            return np.isfinite(c) and c >= conf_thres
        return True

    # Draw joints first.
    for i in range(n):
        if not valid_idx(i):
            continue
        x, y = int(round(float(kpts[i, 0]))), int(round(float(kpts[i, 1])))
        cv2.circle(image, (x, y), joint_radius, joint_color, -1, lineType=cv2.LINE_AA)

    # Draw skeleton connections.
    for i, j in COCO17_EDGES:
        if not (valid_idx(i) and valid_idx(j)):
            continue
        p1 = (int(round(float(kpts[i, 0]))), int(round(float(kpts[i, 1]))))
        p2 = (int(round(float(kpts[j, 0]))), int(round(float(kpts[j, 1]))))
        cv2.line(image, p1, p2, edge_color, edge_thickness, lineType=cv2.LINE_AA)


def extract_pose_instances(result, conf_thres: float = 0.01) -> list[np.ndarray]:
    """Extract pose instances as ``(K,3)`` arrays [x, y, conf] from one Ultralytics result."""
    if not hasattr(result, "keypoints") or result.keypoints is None:
        return []

    try:
        xy = result.keypoints.xy.cpu().numpy()      # (N,K,2) absolute pixel coords
    except Exception:
        return []

    try:
        conf = result.keypoints.conf
        conf_np = conf.cpu().numpy() if conf is not None else None  # (N,K) or None
    except Exception:
        conf_np = None

    poses: list[np.ndarray] = []
    for i in range(len(xy)):
        pts = np.asarray(xy[i], dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 2:
            continue

        if conf_np is None:
            c = np.ones((pts.shape[0], 1), dtype=np.float32)
        else:
            cvals = np.asarray(conf_np[i], dtype=np.float32).reshape(-1, 1)
            if cvals.shape[0] != pts.shape[0]:
                # Fallback if dimensions mismatch.
                cvals = np.ones((pts.shape[0], 1), dtype=np.float32)
            c = cvals

        kpts = np.concatenate([pts, c], axis=1)  # (K,3)
        # Keep instances with at least one reasonably confident point.
        if np.any(np.isfinite(kpts[:, :2])) and np.any(kpts[:, 2] >= conf_thres):
            poses.append(kpts)

    return poses


def _sanitize_kpt_entry(raw: np.ndarray | list) -> np.ndarray | None:
    """Convert a tracker keypoint item into a stable ``(K,2|3)`` numeric array."""
    try:
        arr = np.asarray(raw, dtype=np.float32)
    except Exception:
        return None

    # Some outputs can be object/ragged; convert best-effort to a dense array.
    if arr.dtype == object:
        try:
            arr = np.array(list(arr), dtype=np.float32)
        except Exception:
            return None

    if arr.ndim == 2:
        pass
    elif arr.ndim > 2:
        # Flatten leading dims, then use first pose instance.
        try:
            arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])[0]
        except Exception:
            return None
    else:
        return None

    if arr.shape[0] == 0 or arr.shape[1] < 2:
        return None

    return arr


def _safe_stack_kpt_sequence(seq: list[np.ndarray]) -> np.ndarray | None:
    """Stack keypoint frames with shape checks and fallback trimming."""
    clean: list[np.ndarray] = []
    for raw in seq:
        arr = _sanitize_kpt_entry(raw)
        if arr is not None:
            clean.append(arr)
    if not clean:
        return None

    # Keep only entries with the most common shape.
    shape_counts: dict[tuple[int, int], int] = {}
    for arr in clean:
        key = (arr.shape[0], arr.shape[1])
        shape_counts[key] = shape_counts.get(key, 0) + 1
    target_shape = max(shape_counts, key=shape_counts.get)
    filtered = [arr for arr in clean if (arr.shape[0], arr.shape[1]) == target_shape]
    if not filtered:
        return None

    try:
        return np.stack(filtered).astype(np.float32)
    except Exception:
        return None


def _normalize_key(text: str) -> str:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", (text or "").strip())
    normalized = normalized.replace("-", " ").replace("_", " ")
    return "_".join(normalized.lower().split())


def _split_reference_key(reference_key: str) -> tuple[str, str | None]:
    """Parse `<technique>` or `<technique>__<angle>` reference key."""
    key = (reference_key or "").strip()
    if "__" in key:
        technique, angle = key.split("__", 1)
        return _normalize_key(technique), _normalize_key(angle)
    return _normalize_key(key), None


def _coerce_reference_array(arr: np.ndarray) -> np.ndarray | None:
    """Normalize stored reference shape to `(T,K,D)` where D>=2."""
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim > 3:
        try:
            arr = arr.reshape(-1, arr.shape[-3], arr.shape[-2], arr.shape[-1])[0]
        except Exception:
            return None

    if arr.ndim != 3 or arr.shape[-1] < 2:
        return None
    return arr.astype(np.float32)


def _put_reference(
    refs: dict[str, dict[str, np.ndarray]], technique: str, angle: str, sequence: np.ndarray
) -> None:
    t = _normalize_key(technique)
    a = _normalize_key(angle)
    if not t or not a:
        return
    refs.setdefault(t, {})[a] = sequence.astype(np.float32)


def load_reference_pose_library(reference_dir: str) -> dict[str, dict[str, np.ndarray]]:
    """Load canonical technique sequences from `reference_dir`.

    Supports both layouts:
    - Legacy flat: `reference_dir/<technique>.npy`
    - Multi-angle: `reference_dir/<technique>/<angle>.npy`
    """
    path = Path(reference_dir)
    if not path.exists():
        return {}

    refs: dict[str, dict[str, np.ndarray]] = {}

    # Legacy flat files remain supported as a default angle.
    for fp in path.glob("*.npy"):
        try:
            arr = np.load(fp)
        except Exception:
            continue
        seq = _coerce_reference_array(arr)
        if seq is None:
            continue
        _put_reference(refs, technique=fp.stem, angle="default", sequence=seq)

    # New multi-angle structure: reference_dir/<technique>/<angle>.npy
    for technique_dir in path.iterdir():
        if not technique_dir.is_dir():
            continue
        for fp in technique_dir.glob("*.npy"):
            try:
                arr = np.load(fp)
            except Exception:
                continue
            seq = _coerce_reference_array(arr)
            if seq is None:
                continue
            _put_reference(refs, technique=technique_dir.name, angle=fp.stem, sequence=seq)

    return refs


def _next_indexed_reference_path(out_dir: Path, stem: str) -> Path:
    indices: list[int] = []
    plain_path = out_dir / f"{stem}.npy"
    if plain_path.exists():
        indices.append(1)
    for fp in out_dir.glob(f"{stem}_*.npy"):
        suffix = fp.stem[len(stem) + 1 :]
        if suffix.isdigit():
            indices.append(int(suffix))
    next_index = (max(indices) + 1) if indices else 1
    return out_dir / f"{stem}_{next_index:02d}.npy"


def save_reference_pose(
    reference_dir: str, technique: str, sequence: np.ndarray, append_indexed: bool = False
) -> Path:
    """Store a canonical pose sequence.

    Accepted `technique` forms:
    - `<technique>`: legacy flat file at `reference_dir/<technique>.npy`
    - `<technique>__<angle>`: multi-angle file at `reference_dir/<technique>/<angle>.npy`
    """
    path = Path(reference_dir)
    path.mkdir(parents=True, exist_ok=True)
    technique_key, angle_key = _split_reference_key(technique)
    if append_indexed:
        out_dir = path / technique_key
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = angle_key or "default"
        out_path = _next_indexed_reference_path(out_dir, stem)
    else:
        if angle_key:
            out_dir = path / technique_key
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{angle_key}.npy"
        else:
            out_path = path / f"{technique_key}.npy"
    np.save(out_path, sequence.astype(np.float32))
    return out_path


def _best_reference_match(
    user_sequence: np.ndarray,
    reference_bank: dict[str, np.ndarray],
    technique: str,
    conf_thresh: float = 0.2,
) -> tuple[str, dict[str, float | bool]] | None:
    """Return the best-matching angle and metrics from a technique's reference bank."""
    best_angle = ""
    best_metrics: dict[str, float | bool] | None = None
    best_score = -1.0

    for angle, ref_seq in reference_bank.items():
        metrics = compare_pose_sequence(
            user_sequence=user_sequence,
            reference_sequence=ref_seq,
            technique=technique,
            conf_thresh=conf_thresh,
        )
        score = float(metrics["score"])
        if score > best_score:
            best_score = score
            best_angle = angle
            best_metrics = metrics

    if best_metrics is None:
        return None
    return best_angle, best_metrics


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _init_structured_run_storage(storage_root: str, run_name: str | None, run_config: dict) -> dict[str, Path | str]:
    """Create run artifact folders and write initial config metadata."""
    run_id = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    root = Path(storage_root)
    run_dir = root / "runs" / run_id
    tracks_dir = run_dir / "tracks"
    overlays_dir = run_dir / "overlays"
    run_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    payload = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": run_config,
    }
    _write_json(config_path, payload)

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "tracks_dir": tracks_dir,
        "overlays_dir": overlays_dir,
        "metrics_path": run_dir / "metrics.csv",
        "config_path": config_path,
        "storage_root": root,
    }


def _append_metric_row(metrics_path: Path, row: dict) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not metrics_path.exists()
    with metrics_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_reference_meta(reference_path: Path, technique: str, source: str, num_frames: int) -> None:
    meta = {
        "technique": technique,
        "reference_file": str(reference_path),
        "source": source,
        "num_frames": num_frames,
        "keypoint_format": "COCO17",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(reference_path.with_name(reference_path.stem + "_meta.json"), meta)


def _write_track_summary(track_kpts_history: dict[int, list[np.ndarray]], out_path: Path) -> None:
    tracks = []
    for tid, seq in track_kpts_history.items():
        stacked = _safe_stack_kpt_sequence(seq)
        if stacked is None:
            tracks.append({"track_id": float(tid), "frames": len(seq), "shape": None, "stacked": False})
        else:
            tracks.append(
                {
                    "track_id": float(tid),
                    "frames": int(stacked.shape[0]),
                    "shape": [int(x) for x in stacked.shape],
                    "stacked": True,
                }
            )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tracks": tracks,
    }
    _write_json(out_path, payload)


def _load_label_thresholds(path: str | None) -> dict[str, float]:
    """Load per-technique score thresholds from JSON.

    Accepts either:
    - flat mapping: {"jab": 62.0, "cross": 64.0}
    - structured: {"thresholds": {"jab": 62.0, ...}, ...}
    """
    if not path:
        return {}

    p = Path(path)
    if not p.exists():
        print(f"warning: label thresholds file not found: {p}")
        return {}

    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"warning: failed to read label thresholds from {p}: {exc}")
        return {}

    raw = payload.get("thresholds", payload) if isinstance(payload, dict) else {}
    if not isinstance(raw, dict):
        print(f"warning: invalid label threshold format in {p}")
        return {}

    out: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        try:
            out[_normalize_key(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _valid_point(pt: np.ndarray) -> bool:
    return np.isfinite(pt).all()


def _safe_joint(frame: np.ndarray, idx: int) -> np.ndarray | None:
    if idx >= frame.shape[0]:
        return None
    p = frame[idx]
    if p.shape[0] < 2:
        return None
    if not _valid_point(p[:2]):
        return None
    return p[:2]


def _joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle ABC in degrees."""
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6:
        return float("nan")
    cosang = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _resample_1d(series: np.ndarray, target_len: int) -> np.ndarray:
    src_len = len(series)
    if src_len == target_len:
        return series.astype(np.float32)
    if src_len == 1:
        return np.full((target_len,), float(series[0]), dtype=np.float32)
    src_x = np.linspace(0.0, 1.0, src_len)
    dst_x = np.linspace(0.0, 1.0, target_len)
    return np.interp(dst_x, src_x, series).astype(np.float32)


def resample_pose_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a pose sequence ``(T,K,2)`` to ``target_len`` while handling NaNs."""
    t, k, d = seq.shape
    out = np.zeros((target_len, k, d), dtype=np.float32)
    for j in range(k):
        for c in range(d):
            y = seq[:, j, c]
            valid = np.isfinite(y)
            if not valid.any():
                out[:, j, c] = 0.0
                continue
            if valid.sum() == 1:
                out[:, j, c] = float(y[valid][0])
                continue
            y_fill = y.copy()
            valid_idx = np.where(valid)[0]
            first_idx, last_idx = valid_idx[0], valid_idx[-1]
            y_fill[:first_idx] = y_fill[first_idx]
            y_fill[last_idx + 1 :] = y_fill[last_idx]
            miss = ~valid
            if miss.any():
                y_fill[miss] = np.interp(np.where(miss)[0], valid_idx, y_fill[valid_idx])
            out[:, j, c] = _resample_1d(y_fill, target_len)
    return out


def normalize_pose_frame(frame_kpts: np.ndarray, conf_thresh: float = 0.2) -> np.ndarray:
    """Normalize one frame to reduce scale/translation effects.

    Returns shape ``(K,2)`` in a body-centric coordinate system.
    """
    if frame_kpts.ndim != 2 or frame_kpts.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float32)

    xy = frame_kpts[:, :2].astype(np.float32).copy()
    if frame_kpts.shape[1] >= 3:
        conf = frame_kpts[:, 2]
        xy[conf < conf_thresh] = np.nan

    # Center: prefer hip midpoint, else mean of valid points.
    lhip = _safe_joint(xy, 11)
    rhip = _safe_joint(xy, 12)
    if lhip is not None and rhip is not None:
        center = (lhip + rhip) / 2.0
    else:
        valid = np.isfinite(xy).all(axis=1)
        center = xy[valid].mean(axis=0) if valid.any() else np.array([0.0, 0.0], dtype=np.float32)
    xy = xy - center

    # Scale: shoulder distance, then hip distance, then median radial distance.
    lsho = _safe_joint(xy, 5)
    rsho = _safe_joint(xy, 6)
    scale = 0.0
    if lsho is not None and rsho is not None:
        scale = float(np.linalg.norm(lsho - rsho))
    if scale < 1e-6 and lhip is not None and rhip is not None:
        scale = float(np.linalg.norm(lhip - rhip))
    if scale < 1e-6:
        valid = np.isfinite(xy).all(axis=1)
        if valid.any():
            scale = float(np.median(np.linalg.norm(xy[valid], axis=1)))
    if scale < 1e-6:
        scale = 1.0
    xy = xy / scale
    return xy


def normalize_pose_sequence(sequence: np.ndarray, conf_thresh: float = 0.2) -> np.ndarray:
    """Normalize sequence with shape ``(T,K,2|3)`` to ``(T,K,2)``."""
    return np.stack([normalize_pose_frame(frame, conf_thresh=conf_thresh) for frame in sequence]).astype(np.float32)


def _frame_pose_distance(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if not valid.any():
        return 1.0
    return float(np.mean(np.linalg.norm(a[valid] - b[valid], axis=1)))


def dtw_pose_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """DTW distance between two normalized pose sequences ``(T,K,2)``."""
    na, nb = seq_a.shape[0], seq_b.shape[0]
    dp = np.full((na + 1, nb + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = _frame_pose_distance(seq_a[i - 1], seq_b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[na, nb] / max(na, nb))


def _mean_angle_sequence(seq: np.ndarray, a: int, b: int, c: int) -> float:
    vals = []
    for frame in seq:
        pa = _safe_joint(frame, a)
        pb = _safe_joint(frame, b)
        pc = _safe_joint(frame, c)
        if pa is None or pb is None or pc is None:
            continue
        ang = _joint_angle_deg(pa, pb, pc)
        if np.isfinite(ang):
            vals.append(ang)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def technique_angle_error(user_seq: np.ndarray, ref_seq: np.ndarray, technique: str) -> float:
    """Mean absolute angle error for technique-relevant joints."""
    t = technique.lower().replace("-", " ").strip()
    if "jab" in t or "cross" in t or "hook" in t:
        angle_defs = [(5, 7, 9), (6, 8, 10), (7, 5, 11), (8, 6, 12)]
    elif "kick" in t:
        angle_defs = [(11, 13, 15), (12, 14, 16), (5, 11, 13), (6, 12, 14)]
    else:
        angle_defs = [(5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16)]

    errs = []
    for a, b, c in angle_defs:
        u = _mean_angle_sequence(user_seq, a, b, c)
        r = _mean_angle_sequence(ref_seq, a, b, c)
        if np.isfinite(u) and np.isfinite(r):
            errs.append(abs(u - r))
    if not errs:
        return 90.0
    return float(np.mean(errs))


def cosine_pose_similarity(user_seq: np.ndarray, ref_seq: np.ndarray) -> float:
    """Cosine similarity in [-1,1] between flattened normalized sequences."""
    u = np.nan_to_num(user_seq, nan=0.0).reshape(-1)
    r = np.nan_to_num(ref_seq, nan=0.0).reshape(-1)
    nu = np.linalg.norm(u)
    nr = np.linalg.norm(r)
    if nu < 1e-6 or nr < 1e-6:
        return 0.0
    return float(np.dot(u, r) / (nu * nr))


def _mirror_sequence(seq: np.ndarray) -> np.ndarray:
    mirrored = seq.copy()
    mirrored[..., 0] *= -1.0
    return mirrored


def _wrist_motion_energy(seq: np.ndarray, conf_thresh: float = 0.2) -> float:
    """Maximum wrist displacement across a normalized pose sequence.

    Measures how far each wrist (joints 9 = L_wrist, 10 = R_wrist) travels
    from its first detected position throughout the sequence.  Units are in
    shoulder-width (≈1.0), so a value of 0.3 means the wrist moved at least
    30 % of shoulder width — the minimum expected for a real strike.

    Returns the larger of the two wrists' peak displacements.
    """
    norm = normalize_pose_sequence(seq, conf_thresh=conf_thresh)
    max_disp = 0.0
    for wrist_idx in (9, 10):  # L_wrist, R_wrist
        positions = []
        for frame in norm:
            p = _safe_joint(frame, wrist_idx)
            if p is not None:
                positions.append(p)
        if len(positions) >= 2:
            pts = np.array(positions)
            disp = float(np.max(np.linalg.norm(pts - pts[0], axis=1)))
            max_disp = max(max_disp, disp)
    return max_disp


def _wrist_return_closure(seq: np.ndarray, conf_thresh: float = 0.2) -> float:
    """Return-closure score in [0,1] for wrist start vs end positions.

    Higher values indicate the sequence likely includes both extension and
    retraction phases (the wrist returns toward where it started).
    """
    norm = normalize_pose_sequence(seq, conf_thresh=conf_thresh)
    closures: list[float] = []
    for wrist_idx in (9, 10):  # L_wrist, R_wrist
        positions = []
        for frame in norm:
            p = _safe_joint(frame, wrist_idx)
            if p is not None:
                positions.append(p)
        if len(positions) < 2:
            continue

        start = positions[0]
        end = positions[-1]
        end_dist = float(np.linalg.norm(end - start))
        closure = 1.0 - min(1.0, end_dist / 0.6)  # 0.6 shoulder widths ~= no return
        closures.append(float(np.clip(closure, 0.0, 1.0)))

    if not closures:
        return 0.0
    return float(max(closures))


def _extract_event_centered_window(
    seq: np.ndarray,
    window_len: int,
    conf_thresh: float = 0.2,
) -> np.ndarray | None:
    """Extract a fixed-length window centered on peak wrist extension.

    Peak is computed from the wrist (left/right) that has the largest
    displacement from its first valid point in normalized coordinates.
    """
    if seq.ndim != 3 or window_len <= 0 or seq.shape[0] < window_len:
        return None

    norm = normalize_pose_sequence(seq, conf_thresh=conf_thresh)
    total_frames = int(norm.shape[0])
    best_peak_idx: int | None = None
    best_peak_disp = -1.0

    for wrist_idx in (9, 10):
        start_pt: np.ndarray | None = None
        disp = np.full((total_frames,), np.nan, dtype=np.float32)
        for frame_idx in range(total_frames):
            p = _safe_joint(norm[frame_idx], wrist_idx)
            if p is None:
                continue
            if start_pt is None:
                start_pt = p
            disp[frame_idx] = float(np.linalg.norm(p - start_pt))

        if not np.isfinite(disp).any():
            continue

        wrist_peak_idx = int(np.nanargmax(disp))
        wrist_peak_disp = float(disp[wrist_peak_idx])
        if wrist_peak_disp > best_peak_disp:
            best_peak_disp = wrist_peak_disp
            best_peak_idx = wrist_peak_idx

    if best_peak_idx is None:
        return None

    half = window_len // 2
    start = best_peak_idx - half
    start = max(0, min(start, total_frames - window_len))
    end = start + window_len
    return seq[start:end].copy()


def _extract_stance_cycle_sequence(
    seq: np.ndarray,
    start_threshold: float = 0.18,
    end_threshold: float = 0.12,
    peak_threshold: float = 0.30,
    min_frames: int = 24,
    hold_frames: int = 4,
    conf_thresh: float = 0.2,
) -> np.ndarray | None:
    """Extract dynamic start/end from stance departure and return.

    A sequence starts when normalized pose distance from initial stance rises
    above baseline and ends after it settles back near the initial stance for a
    short hold period.
    """
    if seq.ndim != 3 or seq.shape[0] < max(2, min_frames):
        return None

    norm = normalize_pose_sequence(seq, conf_thresh=conf_thresh)
    total = int(norm.shape[0])
    anchor = norm[0]
    dist = np.array([_frame_pose_distance(frame, anchor) for frame in norm], dtype=np.float32)

    if total >= 5:
        kernel = min(9, total if total % 2 == 1 else total - 1)
        if kernel >= 3:
            w = np.ones((kernel,), dtype=np.float32) / float(kernel)
            dist_smooth = np.convolve(dist, w, mode="same").astype(np.float32)
        else:
            dist_smooth = dist
    else:
        dist_smooth = dist

    peak_idx = int(np.argmax(dist_smooth))
    peak_val = float(dist_smooth[peak_idx])
    if peak_val < peak_threshold:
        return None

    start_idx = 0
    for i in range(peak_idx, -1, -1):
        if float(dist_smooth[i]) <= start_threshold:
            start_idx = i
            break

    end_idx: int | None = None
    hold = max(1, int(hold_frames))
    for i in range(peak_idx + 1, total - hold + 1):
        if np.all(dist_smooth[i : i + hold] <= end_threshold):
            end_idx = i + hold - 1
            break

    if end_idx is None:
        trailing = np.where(dist_smooth[peak_idx + 1 :] <= end_threshold)[0]
        if trailing.size > 0:
            end_idx = int(peak_idx + 1 + trailing[-1])
        else:
            return None

    if end_idx <= start_idx or peak_idx < start_idx or peak_idx > end_idx:
        return None

    if (end_idx - start_idx + 1) < int(min_frames):
        return None

    return seq[start_idx : end_idx + 1].copy()


def _passes_reference_score_gate(
    seq: np.ndarray,
    references: dict[str, dict[str, np.ndarray]],
    technique: str,
    min_score: float,
) -> tuple[bool, float]:
    """Check candidate reference against already-stored references.

    Returns ``(passes, best_score)``.  If no reference has been saved yet for
    this technique the gate is bypassed (returns ``True, 0.0``) so the very
    first capture can always be accepted — it establishes the baseline.
    """
    tkey = _normalize_key(technique)
    if tkey not in references or not references[tkey]:
        return True, 0.0  # no existing reference — bypass gate
    best = _best_reference_match(seq, references[tkey], tkey)
    if best is None:
        return False, 0.0
    score = float(best[1]["score"])
    return score >= min_score, score


def _passes_reference_similarity_band(
    seq: np.ndarray,
    references: dict[str, dict[str, np.ndarray]],
    technique: str,
    min_score: float,
    max_score: float,
    bypass_if_missing: bool,
) -> tuple[bool, float, bool]:
    """Check whether a candidate falls inside a reference similarity band.

    Returns ``(passes, best_score, has_reference)``.
    """
    tkey = _normalize_key(technique)
    has_reference = tkey in references and bool(references[tkey])
    if not has_reference:
        return bypass_if_missing, 0.0, False

    best = _best_reference_match(seq, references[tkey], tkey)
    if best is None:
        return False, 0.0, True

    score = float(best[1]["score"])
    return min_score <= score <= max_score, score, True


def _try_record_reference_sequence(
    pose_seq: np.ndarray,
    record_reference: str,
    reference_dir: str,
    references: dict[str, dict[str, np.ndarray]],
    capture_seed_references: dict[str, dict[str, np.ndarray]],
    source: str,
    frame_counter: int,
    ref_min_motion_energy: float,
    ref_min_return_closure: float,
    ref_min_score_gate: float,
    capture_seed_min_score: float,
    capture_seed_max_score: float,
    last_saved_frame: int,
    reference_capture_cooldown_frames: int,
    append_indexed: bool,
    enable_structured_storage: bool,
) -> tuple[bool, int]:
    if reference_capture_cooldown_frames > 0 and last_saved_frame >= 0:
        frames_since_save = frame_counter - last_saved_frame
        if frames_since_save < reference_capture_cooldown_frames:
            return False, last_saved_frame

    energy = _wrist_motion_energy(pose_seq)
    if energy < ref_min_motion_energy:
        print(
            f"  [ref gate] skip: wrist motion {energy:.3f} < {ref_min_motion_energy} — waiting for active sequence"
        )
        return False, last_saved_frame

    closure = _wrist_return_closure(pose_seq)
    if closure < ref_min_return_closure:
        print(
            f"  [ref gate] skip: return closure {closure:.3f} < {ref_min_return_closure} "
            "— sequence likely misses retraction"
        )
        return False, last_saved_frame

    technique_key, angle_key = _split_reference_key(record_reference)
    has_existing_refs = technique_key in references and bool(references[technique_key])
    passes_gate, gate_score = _passes_reference_score_gate(
        pose_seq, references, technique_key, ref_min_score_gate
    )
    if not passes_gate:
        print(
            f"  [ref gate] skip: score {gate_score:.1f}/100 < {ref_min_score_gate} — sequence doesn't match technique"
        )
        return False, last_saved_frame

    passes_seed_gate, seed_score, has_seed_refs = _passes_reference_similarity_band(
        pose_seq,
        capture_seed_references,
        technique_key,
        capture_seed_min_score,
        capture_seed_max_score,
        bypass_if_missing=True,
    )
    if not passes_seed_gate:
        if has_seed_refs:
            print(
                f"  [seed gate] skip: score {seed_score:.1f}/100 not in "
                f"[{capture_seed_min_score:.1f}, {capture_seed_max_score:.1f}] — candidate is either too far from or too close to the seed reference"
            )
        else:
            print(
                f"  [seed gate] skip: no seed references found for technique '{technique_key}' in capture seed bank"
            )
        return False, last_saved_frame

    ref_path = save_reference_pose(
        reference_dir,
        record_reference,
        pose_seq,
        append_indexed=append_indexed,
    )
    stored_angle = ref_path.stem if ref_path.parent.name == technique_key else (angle_key or "default")
    _put_reference(refs=references, technique=technique_key, angle=stored_angle, sequence=pose_seq)

    gate_text = f"{gate_score:.1f}" if has_existing_refs else "bootstrap"
    seed_text = f", seed={seed_score:.1f}" if has_seed_refs else ""
    print(
        f"saved reference technique '{record_reference}' to: {ref_path} "
        f"(motion={energy:.3f}, closure={closure:.3f}, gate={gate_text}{seed_text}, frame={frame_counter})"
    )
    if enable_structured_storage:
        _write_reference_meta(
            reference_path=ref_path,
            technique=record_reference,
            source=str(source),
            num_frames=int(pose_seq.shape[0]),
        )
    return True, frame_counter


def compare_pose_sequence(
    user_sequence: np.ndarray,
    reference_sequence: np.ndarray,
    technique: str,
    conf_thresh: float = 0.2,
) -> dict[str, float | bool]:
    """Compare user sequence against reference and return metric bundle."""
    ref_norm = normalize_pose_sequence(reference_sequence, conf_thresh=conf_thresh)
    user_norm = normalize_pose_sequence(user_sequence, conf_thresh=conf_thresh)

    target_len = max(4, ref_norm.shape[0])
    ref_res = resample_pose_sequence(ref_norm, target_len)
    usr_res = resample_pose_sequence(user_norm, target_len)
    usr_mirror_res = _mirror_sequence(usr_res)

    # Choose orientation that best matches reference.
    cos_plain = cosine_pose_similarity(usr_res, ref_res)
    cos_mirror = cosine_pose_similarity(usr_mirror_res, ref_res)
    use_mirror = cos_mirror > cos_plain
    usr_best = usr_mirror_res if use_mirror else usr_res
    cos_sim = max(cos_plain, cos_mirror)

    dtw_dist = dtw_pose_distance(usr_best, ref_res)
    angle_err = technique_angle_error(usr_best, ref_res, technique)
    mean_dist = float(np.mean([_frame_pose_distance(a, b) for a, b in zip(usr_best, ref_res)]))

    # Convert metrics to [0,100] then combine.
    cosine_score = (cos_sim + 1.0) * 50.0
    dtw_score = max(0.0, 100.0 * (1.0 - (dtw_dist / 0.8)))
    angle_score = max(0.0, 100.0 * (1.0 - (angle_err / 90.0)))
    pose_dist_score = max(0.0, 100.0 * (1.0 - (mean_dist / 0.8)))
    final_score = 0.35 * cosine_score + 0.25 * dtw_score + 0.25 * angle_score + 0.15 * pose_dist_score

    return {
        "use_mirror": use_mirror,
        "cosine_similarity": float(cos_sim),
        "dtw_distance": float(dtw_dist),
        "angle_error": float(angle_err),
        "mean_pose_distance": float(mean_dist),
        "score": float(np.clip(final_score, 0.0, 100.0)),
    }


def generate_feedback(technique: str, user_sequence: np.ndarray, score: float) -> list[str]:
    """Generate concise feedback strings from normalized user sequence."""
    t = technique.lower().replace("-", " ").strip()
    seq = normalize_pose_sequence(user_sequence)
    if len(seq) == 0:
        return ["No pose detected"]

    msgs: list[str] = []
    # Use final frame for quick rule checks.
    f = seq[-1]

    def angle(a: int, b: int, c: int) -> float:
        pa, pb, pc = _safe_joint(f, a), _safe_joint(f, b), _safe_joint(f, c)
        if pa is None or pb is None or pc is None:
            return float("nan")
        return _joint_angle_deg(pa, pb, pc)

    if "jab" in t or "cross" in t or "hook" in t:
        lelbow = angle(5, 7, 9)
        relbow = angle(6, 8, 10)
        if np.isfinite(lelbow) and np.isfinite(relbow):
            punch_arm_right = relbow > lelbow
            punch_elbow = relbow if punch_arm_right else lelbow
            guard_wrist = _safe_joint(f, 9 if punch_arm_right else 10)
            guard_shoulder = _safe_joint(f, 5 if punch_arm_right else 6)
            if punch_elbow < 150:
                msgs.append("Extend your punching arm more")
            if guard_wrist is not None and guard_shoulder is not None and guard_wrist[1] > guard_shoulder[1] + 0.15:
                msgs.append("Keep your guard hand higher")
    elif "kick" in t:
        lknee = angle(11, 13, 15)
        rknee = angle(12, 14, 16)
        if np.isfinite(lknee) and np.isfinite(rknee):
            kick_right = rknee > lknee
            kick_knee = rknee if kick_right else lknee
            support_knee = lknee if kick_right else rknee
            if kick_knee < 155:
                msgs.append("Extend the kicking leg more")
            if support_knee < 145:
                msgs.append("Stabilize and straighten your support leg")
    if score < 60:
        msgs.append("Slow down and focus on form")
    if not msgs:
        msgs.append("Good form")
    return msgs[:2]


def _compact_text(text: str, max_chars: int = 52) -> str:
    """Return a compact single-line string for overlays."""
    cleaned = " ".join(str(text).strip().split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1] + "…"


def _pose_center_and_scale(frame_kpts: np.ndarray, conf_thresh: float = 0.2) -> tuple[np.ndarray, float]:
    """Return body anchor center and scale in image coordinates."""
    if frame_kpts.ndim != 2 or frame_kpts.shape[1] < 2:
        return np.array([0.0, 0.0], dtype=np.float32), 1.0

    xy = frame_kpts[:, :2].astype(np.float32)
    conf = frame_kpts[:, 2] if frame_kpts.shape[1] >= 3 else np.ones((frame_kpts.shape[0],), dtype=np.float32)
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(conf) & (conf >= conf_thresh)

    def get_joint(idx: int) -> np.ndarray | None:
        if idx >= xy.shape[0] or not valid[idx]:
            return None
        return xy[idx]

    lhip = get_joint(11)
    rhip = get_joint(12)
    if lhip is not None and rhip is not None:
        center = (lhip + rhip) / 2.0
    else:
        center = xy[valid].mean(axis=0) if valid.any() else np.array([0.0, 0.0], dtype=np.float32)

    lsho = get_joint(5)
    rsho = get_joint(6)
    scale = 0.0
    if lsho is not None and rsho is not None:
        scale = float(np.linalg.norm(lsho - rsho))
    if scale < 1e-6 and lhip is not None and rhip is not None:
        scale = float(np.linalg.norm(lhip - rhip))
    if scale < 1e-6 and valid.any():
        scale = float(np.median(np.linalg.norm(xy[valid] - center, axis=1)))
    return center.astype(np.float32), max(scale, 1.0)


def _technique_focus_joint_indices(technique: str) -> list[int]:
    """Return joints that matter most for visual correction arrows."""
    t = _normalize_key(technique)
    if "kick" in t:
        return [11, 12, 13, 14, 15, 16]
    if t in {"jab", "cross", "hook", "uppercut", "elbow_strike"}:
        return [5, 6, 7, 8, 9, 10]
    return [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def _project_reference_frame_to_image(
    reference_frame_norm: np.ndarray,
    current_frame_kpts: np.ndarray,
    conf_thresh: float = 0.2,
) -> np.ndarray | None:
    """Map a normalized reference frame back into the user's current image frame."""
    if reference_frame_norm.ndim != 2 or reference_frame_norm.shape[1] < 2:
        return None

    center, scale = _pose_center_and_scale(current_frame_kpts, conf_thresh=conf_thresh)
    projected = np.zeros((reference_frame_norm.shape[0], 3), dtype=np.float32)
    projected[:, :2] = (reference_frame_norm[:, :2] * scale) + center
    projected[:, 2] = 1.0
    invalid = ~np.isfinite(reference_frame_norm[:, :2]).all(axis=1)
    projected[invalid, :2] = np.nan
    projected[invalid, 2] = 0.0
    return projected


def _build_reference_overlay(
    user_sequence: np.ndarray,
    reference_sequence: np.ndarray,
    current_frame_kpts: np.ndarray,
    technique: str,
    use_mirror: bool,
    conf_thresh: float = 0.2,
) -> tuple[np.ndarray | None, list[tuple[tuple[int, int], tuple[int, int]]]]:
    """Build a projected ghost pose and top correction arrows for the current frame."""
    ref_norm = normalize_pose_sequence(reference_sequence, conf_thresh=conf_thresh)
    user_norm = normalize_pose_sequence(user_sequence, conf_thresh=conf_thresh)
    current_norm = normalize_pose_frame(current_frame_kpts, conf_thresh=conf_thresh)
    if ref_norm.size == 0 or user_norm.size == 0 or current_norm.size == 0:
        return None, []

    target_len = max(4, ref_norm.shape[0])
    ref_res = resample_pose_sequence(ref_norm, target_len)
    user_res = resample_pose_sequence(user_norm, target_len)
    ref_best = _mirror_sequence(ref_res) if use_mirror else ref_res
    user_last = current_norm

    best_idx = 0
    best_dist = float("inf")
    for idx, ref_frame in enumerate(ref_best):
        dist = _frame_pose_distance(user_last, ref_frame)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx

    projected = _project_reference_frame_to_image(ref_best[best_idx], current_frame_kpts, conf_thresh=conf_thresh)
    if projected is None:
        return None, []

    focus_joints = _technique_focus_joint_indices(technique)
    ranked_errors: list[tuple[float, tuple[tuple[int, int], tuple[int, int]]]] = []
    for joint_idx in focus_joints:
        current_joint_norm = _safe_joint(user_last, joint_idx)
        ref_joint_norm = _safe_joint(ref_best[best_idx], joint_idx)
        if current_joint_norm is None or ref_joint_norm is None:
            continue
        if joint_idx >= current_frame_kpts.shape[0] or joint_idx >= projected.shape[0]:
            continue
        curr_xy = current_frame_kpts[joint_idx, :2]
        tgt_xy = projected[joint_idx, :2]
        if not (np.isfinite(curr_xy).all() and np.isfinite(tgt_xy).all()):
            continue
        diff = float(np.linalg.norm(current_joint_norm - ref_joint_norm))
        ranked_errors.append(
            (
                diff,
                (
                    (int(round(float(curr_xy[0]))), int(round(float(curr_xy[1])))),
                    (int(round(float(tgt_xy[0]))), int(round(float(tgt_xy[1])))),
                ),
            )
        )

    ranked_errors.sort(key=lambda item: item[0], reverse=True)
    arrows = [pair for diff, pair in ranked_errors[:3] if diff >= 0.12]
    return projected, arrows


def _draw_reference_ghost(
    frame: np.ndarray,
    ghost_pose: np.ndarray | None,
) -> None:
    """Render the projected target pose onto the frame."""
    if ghost_pose is not None:
        overlay = frame.copy()
        draw_pose_cv2(
            overlay,
            ghost_pose,
            conf_thres=0.01,
            joint_color=(90, 255, 90),
            edge_color=(40, 220, 40),
            joint_radius=3,
            edge_thickness=2,
        )
        cv2.addWeighted(overlay, 0.42, frame, 0.58, 0, frame)


def _draw_info_panel(
    frame: np.ndarray,
    x: int,
    y: int,
    lines: list[str],
    ok_state: bool,
    font_scale: float = 0.5,
) -> None:
    """Draw a compact semi-transparent label block for trainer text."""
    if not lines:
        return

    line_height = 18
    text_padding_x = 8
    text_padding_y = 8
    max_width = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        max_width = max(max_width, w)

    panel_w = max_width + (2 * text_padding_x)
    panel_h = (len(lines) * line_height) + (2 * text_padding_y)

    h, w = frame.shape[:2]
    x1 = max(5, min(x, w - panel_w - 5))
    y1 = max(5, min(y, h - panel_h - 5))
    x2 = x1 + panel_w
    y2 = y1 + panel_h

    base_color = (36, 140, 36) if ok_state else (24, 24, 156)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), base_color, -1)
    cv2.addWeighted(overlay, 0.36, frame, 0.64, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 1)

    for i, line in enumerate(lines):
        text_y = y1 + text_padding_y + 12 + (i * line_height)
        cv2.putText(
            frame,
            line,
            (x1 + text_padding_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


def evaluate_feedback_quality(
    evaluate_dir: str,
    references: dict[str, dict[str, np.ndarray]],
    technique: str,
    score_threshold: float,
) -> dict[str, float | int] | None:
    """Evaluate scoring quality over saved sequences.

    Expected layout:
    evaluate_dir/<technique>/correct/*.npy
    evaluate_dir/<technique>/incorrect/*.npy
    """
    technique_key = _normalize_key(technique)
    if technique_key not in references:
        return None

    base = Path(evaluate_dir) / technique
    correct_files = sorted((base / "correct").glob("*.npy")) if (base / "correct").exists() else []
    incorrect_files = sorted((base / "incorrect").glob("*.npy")) if (base / "incorrect").exists() else []
    total = len(correct_files) + len(incorrect_files)
    if total == 0:
        return None

    tp = tn = fp = fn = 0
    reference_bank = references[technique_key]
    for fp_seq in correct_files:
        seq = np.load(fp_seq)
        best = _best_reference_match(seq, reference_bank, technique_key)
        pred_correct = best is not None and float(best[1]["score"]) >= score_threshold
        if pred_correct:
            tp += 1
        else:
            fn += 1

    for fp_seq in incorrect_files:
        seq = np.load(fp_seq)
        best = _best_reference_match(seq, reference_bank, technique_key)
        pred_correct = best is not None and float(best[1]["score"]) >= score_threshold
        if pred_correct:
            fp += 1
        else:
            tn += 1

    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "total": total,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def crop_and_pad(frame: np.ndarray, box: list[float], margin_percent: int) -> np.ndarray:
    """Crop box with margin and take square crop from frame.

    Args:
        frame (np.ndarray): The input frame to crop from.
        box (list[float]): The bounding box coordinates [x1, y1, x2, y2].
        margin_percent (int): The percentage of margin to add around the box.

    Returns:
        (np.ndarray): The cropped and resized square image.
    """
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1

    # Add margin
    margin_x, margin_y = int(w * margin_percent / 100), int(h * margin_percent / 100)
    x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    x2, y2 = min(frame.shape[1], x2 + margin_x), min(frame.shape[0], y2 + margin_y)

    # Take square crop from frame
    size = max(y2 - y1, x2 - x1)
    center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
    half_size = size // 2
    square_crop = frame[
        max(0, center_y - half_size) : min(frame.shape[0], center_y + half_size),
        max(0, center_x - half_size) : min(frame.shape[1], center_x + half_size),
    ]

    return cv2.resize(square_crop, (224, 224), interpolation=cv2.INTER_LINEAR)


def _box_motion_energy(box_history: list[np.ndarray], frame_width: int, frame_height: int) -> float:
    """Maximum center displacement for a track's recent boxes, normalized by frame diagonal."""
    if len(box_history) < 2:
        return 0.0
    centers = []
    for box in box_history:
        x1, y1, x2, y2 = box.astype(np.float32)
        centers.append(np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32))
    pts = np.stack(centers)
    diag = max(float(math.hypot(frame_width, frame_height)), 1.0)
    return float(np.max(np.linalg.norm(pts - pts[0], axis=1)) / diag)


def _track_activity_score(
    kpt_history: list[np.ndarray],
    box_history: list[np.ndarray],
    current_box: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> float:
    """Heuristic score for choosing the primary fighter.

    Strongly favors recent wrist motion, with smaller contributions from track motion
    and subject size to break ties when multiple people are visible.
    """
    wrist_energy = 0.0
    if len(kpt_history) >= 2:
        stacked = _safe_stack_kpt_sequence(kpt_history)
        if stacked is not None and len(stacked) >= 2:
            wrist_energy = _wrist_motion_energy(stacked)

    box_energy = _box_motion_energy(box_history, frame_width, frame_height)
    x1, y1, x2, y2 = current_box.astype(np.float32)
    area = max((x2 - x1) * (y2 - y1), 0.0)
    frame_area = max(float(frame_width * frame_height), 1.0)
    area_ratio = float(area / frame_area)
    return 2.5 * wrist_energy + 0.75 * box_energy + 0.15 * math.sqrt(area_ratio)


def _select_primary_track(
    track_ids: list[int],
    track_boxes: dict[int, np.ndarray],
    track_kpts_history: dict[int, list[np.ndarray]],
    track_box_history: dict[int, list[np.ndarray]],
    frame_width: int,
    frame_height: int,
    current_primary_track_id: int | None,
    current_frame: int,
    hold_until_frame: int,
    primary_track_switch_margin: float,
    person_selection_mode: str,
) -> tuple[int | None, int, dict[int, float]]:
    """Return the active track id, hold-until frame, and per-track scores."""
    if not track_ids:
        return None, hold_until_frame, {}

    if person_selection_mode == "all":
        return None, hold_until_frame, {}

    scores: dict[int, float] = {}
    for track_id in track_ids:
        scores[track_id] = _track_activity_score(
            kpt_history=track_kpts_history.get(track_id, []),
            box_history=track_box_history.get(track_id, []),
            current_box=track_boxes[track_id],
            frame_width=frame_width,
            frame_height=frame_height,
        )

    best_track_id = max(scores, key=scores.get)
    best_score = scores[best_track_id]

    if current_primary_track_id in scores:
        current_score = scores[current_primary_track_id]
        if current_frame <= hold_until_frame or current_score * primary_track_switch_margin >= best_score:
            return current_primary_track_id, hold_until_frame, scores

    return best_track_id, current_frame, scores


def run(
    weights: str = "yolo26n-pose.pt",
    device: str = "",
    source: str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    output_path: str | None = None,
    crop_margin_percentage: int = 10,
    num_video_sequence_samples: int = 8,
    skip_frame: int = 1,
    video_cls_overlap_ratio: float = 0.25,
    fp16: bool = False,
    video_classifier_model: str = "microsoft/xclip-base-patch32",
    labels: list[str] | None = None,
    save_kpts_dir: str | None = None,
    visualize_pose: bool = False,
    pose_output_path: str | None = None,
    draw_boxes: bool = True,
    overlay_pose: bool = True,
    reference_dir: str = "reference_poses",
    target_technique: str = "jab",
    trainer_enabled: bool = True,
    record_reference: str | None = None,
    reference_capture_mode: str = "first_valid",
    reference_sequence_mode: str = "fixed",
    reference_capture_buffer_multiplier: int = 3,
    record_reference_max_saves: int = 1,
    reference_capture_cooldown_frames: int = 24,
    auto_exit_after_reference: bool = False,
    reference_search_max_frames: int = 0,
    ref_min_motion_energy: float = 0.3,
    ref_min_return_closure: float = 0.15,
    ref_min_score_gate: float = 75.0,
    capture_seed_reference_dir: str | None = None,
    capture_seed_min_score: float = 0.0,
    capture_seed_max_score: float = 100.0,
    ref_stance_start_threshold: float = 0.18,
    ref_stance_end_threshold: float = 0.12,
    ref_stance_peak_threshold: float = 0.30,
    ref_stance_min_frames: int = 24,
    ref_stance_hold_frames: int = 4,
    person_selection_mode: str = "most_motion",
    primary_track_hold_frames: int = 15,
    primary_track_switch_margin: float = 1.15,
    trainer_score_threshold: float = 70.0,
    evaluate_dir: str | None = None,
    label_thresholds_path: str | None = None,
    enable_video_classifier: bool = True,
    storage_root: str = "data",
    run_name: str | None = None,
    enable_structured_storage: bool = True,
    debug: bool = False,
    display: bool = True,
    fast_mode: bool = False,
) -> None:
    """Run action recognition on a video source using YOLO for object detection (or pose
    estimation) and a video classifier.

    A pose-capable YOLO model (e.g. any YOLO26-pose weights) will produce keypoints which are
    optionally stored and drawn on the output frames.  The rest of the pipeline continues to use
    cropped person patches for action classification as before.

    Args:
        weights (str): Path to the YOLO model weights.  Specifying a "-pose" file will enable
            pose keypoint extraction.
        device (str): Device to run the model on. Use 'cuda' for NVIDIA GPU, 'mps' for Apple Silicon, or 'cpu'.
        source (str): Path to mp4 video file or YouTube URL.
        output_path (str, optional): Path to save the output video.
        crop_margin_percentage (int): Percentage of margin to add around detected objects.
        num_video_sequence_samples (int): Number of video frames to use for classification.
        skip_frame (int): Number of frames to skip between detections.
        video_cls_overlap_ratio (float): Overlap ratio between video sequences.
        fp16 (bool): Whether to use half-precision floating point.
        video_classifier_model (str): Name or path of the video classifier model.
        labels (list[str], optional): List of labels for zero-shot classification.
        save_kpts_dir (str, optional): Directory in which to store per-track keypoint
            sequences.  If specified the folder will be created and each track will be
            written as ``track_<id>.npy`` after processing completes.
        visualize_pose (bool): If True, show a second window containing only the
            skeleton overlay (black background) for each detected person.
        pose_output_path (str, optional): Path to save a video of the pose-only window.
            Only used when ``visualize_pose`` is True.  The file will have the same
            resolution as the input frames.
            If not provided the default ``datasets/pose_video.mp4`` will be used
            (directory is created if necessary).
        draw_boxes (bool): If False, bounding-box annotations will not be drawn on the
            main video; useful when using pose models and you only want skeletons.
        reference_dir (str): Folder containing canonical technique sequences as
            ``<technique>.npy`` files.
        target_technique (str): Technique name to score against, e.g. ``jab`` or
            ``front_kick``.
        trainer_enabled (bool): Enable live pose scoring and feedback.
        record_reference (str, optional): If provided, records a tracked pose sequence
            as a new reference under this technique name.
        reference_capture_mode (str): Reference capture strategy. ``first_valid`` saves
            immediately when a valid window is found. ``best_window`` scans windows
            across the clip and saves the highest-scoring valid candidate at the end.
            ``event_centered`` extracts a window around peak wrist extension from
            the rolling capture buffer and applies the same quality gates.
        reference_sequence_mode (str): Candidate sequence extraction mode.
            ``fixed`` uses the latest fixed-length window,
            ``event_centered`` centers on peak wrist extension,
            ``stance_cycle`` extracts from departure to return to stance.
        reference_capture_buffer_multiplier (int): Multiplier for rolling keypoint
            buffer length during reference capture. Effective history length becomes
            ``num_video_sequence_samples * reference_capture_buffer_multiplier``.
            Ignored when not recording references.
        capture_seed_reference_dir (str, optional): Optional reference directory used
            only for capture gating. This is useful when Golden Seed references should
            anchor new captures without forcing the output to overwrite the seed bank.
        capture_seed_min_score (float): Minimum similarity to the seed bank required
            before a candidate reference is accepted.
        capture_seed_max_score (float): Maximum similarity to the seed bank allowed
            before a candidate is treated as too identical to the seed reference.
        trainer_score_threshold (float): Threshold used by evaluation loop to classify
            correct vs incorrect technique execution.
        evaluate_dir (str, optional): Optional dataset root for offline evaluation.
        label_thresholds_path (str, optional): JSON file containing per-technique
            score thresholds. When set, per-label thresholds override
            ``trainer_score_threshold`` for matching keys.
        enable_video_classifier (bool): If False, skip action-label classifier setup
            and inference. Useful for pose/template capture in offline or restricted
            network environments.
        storage_root (str): Root folder for structured artifacts (runs, references).
        run_name (str, optional): Optional run ID override for artifact folder naming.
        enable_structured_storage (bool): If True, persist run config, metrics and
            track summaries under ``storage_root``.
        fast_mode (bool): Apply speed-optimized defaults for non-visual runs.
    """

    if skip_frame <= 0:
        raise ValueError("--skip-frame must be >= 1")
    if num_video_sequence_samples <= 0:
        raise ValueError("--num-video-sequence-samples must be >= 1")
    if not (0.0 <= video_cls_overlap_ratio < 1.0):
        raise ValueError("--video-cls-overlap-ratio must be in [0.0, 1.0)")
    if reference_capture_mode not in {"first_valid", "best_window", "event_centered"}:
        raise ValueError("--reference-capture-mode must be one of: first_valid, best_window, event_centered")
    if reference_sequence_mode not in {"fixed", "event_centered", "stance_cycle"}:
        raise ValueError("--reference-sequence-mode must be one of: fixed, event_centered, stance_cycle")
    if reference_capture_buffer_multiplier < 1:
        raise ValueError("--reference-capture-buffer-multiplier must be >= 1")
    if capture_seed_min_score < 0.0 or capture_seed_max_score > 100.0:
        raise ValueError("--capture-seed min/max scores must be in [0, 100]")
    if capture_seed_min_score > capture_seed_max_score:
        raise ValueError("--capture-seed-min-score must be <= --capture-seed-max-score")
    if ref_stance_min_frames < 2:
        raise ValueError("--ref-stance-min-frames must be >= 2")
    if ref_stance_hold_frames < 1:
        raise ValueError("--ref-stance-hold-frames must be >= 1")

    if fast_mode:
        # Preset tuned for throughput: disable expensive visualization/classifier paths.
        display = False
        visualize_pose = True  # Keep pose visualization for debugging/tracing even in fast mode.
        draw_boxes = True  # Box drawing is relatively cheap and helps with debugging.
        overlay_pose = True
        enable_video_classifier = False
        # Reference capture currently flows through trainer logic, so only disable
        # trainer when not capturing a reference.
        if not record_reference:
            trainer_enabled = True
        print(
            "fast-mode enabled: "
            f"display={display}, visualize_pose={visualize_pose}, draw_boxes={draw_boxes}, "
            f"overlay_pose={overlay_pose}, video_classifier={enable_video_classifier}, "
            f"trainer={trainer_enabled}, skip_frame={skip_frame}"
        )

    video_cls_step = max(1, int(round(num_video_sequence_samples * skip_frame * (1.0 - video_cls_overlap_ratio))))

    if labels is None or len(labels) == 0:
        labels = MARTIAL_ARTS_LABELS.copy()
    weights = str(_resolve_project_path(weights))
    reference_dir = str(_resolve_project_path(reference_dir))
    if capture_seed_reference_dir:
        capture_seed_reference_dir = str(_resolve_project_path(capture_seed_reference_dir))
    if evaluate_dir:
        evaluate_dir = str(_resolve_project_path(evaluate_dir))
    if label_thresholds_path:
        label_thresholds_path = str(_resolve_project_path(label_thresholds_path))
    storage_root = str(_resolve_project_path(storage_root))
    if output_path:
        output_path = str(_resolve_project_path(output_path))
    if save_kpts_dir:
        save_kpts_dir = str(_resolve_project_path(save_kpts_dir))
    if pose_output_path:
        pose_output_path = str(_resolve_project_path(pose_output_path))

    source_path = Path(source)
    source_is_webcam = source.isdigit()
    source_is_url = source.startswith("http")
    if not source_is_webcam and not source_is_url and source_path.suffix:
        source = str(_resolve_project_path(source))

    run_config = {
        "weights": weights,
        "device": device,
        "source": source,
        "output_path": output_path,
        "crop_margin_percentage": crop_margin_percentage,
        "num_video_sequence_samples": num_video_sequence_samples,
        "skip_frame": skip_frame,
        "video_cls_overlap_ratio": video_cls_overlap_ratio,
        "fp16": fp16,
        "video_classifier_model": video_classifier_model,
        "labels": labels,
        "save_kpts_dir": save_kpts_dir,
        "visualize_pose": visualize_pose,
        "pose_output_path": pose_output_path,
        "draw_boxes": draw_boxes,
        "overlay_pose": overlay_pose,
        "reference_dir": reference_dir,
        "target_technique": target_technique,
        "trainer_enabled": trainer_enabled,
        "record_reference": record_reference,
        "reference_capture_mode": reference_capture_mode,
        "reference_sequence_mode": reference_sequence_mode,
        "reference_capture_buffer_multiplier": reference_capture_buffer_multiplier,
        "ref_min_motion_energy": ref_min_motion_energy,
        "ref_min_return_closure": ref_min_return_closure,
        "ref_min_score_gate": ref_min_score_gate,
        "capture_seed_reference_dir": capture_seed_reference_dir,
        "capture_seed_min_score": capture_seed_min_score,
        "capture_seed_max_score": capture_seed_max_score,
        "ref_stance_start_threshold": ref_stance_start_threshold,
        "ref_stance_end_threshold": ref_stance_end_threshold,
        "ref_stance_peak_threshold": ref_stance_peak_threshold,
        "ref_stance_min_frames": ref_stance_min_frames,
        "ref_stance_hold_frames": ref_stance_hold_frames,
        "trainer_score_threshold": trainer_score_threshold,
        "evaluate_dir": evaluate_dir,
        "label_thresholds_path": label_thresholds_path,
        "enable_video_classifier": enable_video_classifier,
        "storage_root": storage_root,
        "run_name": run_name,
        "enable_structured_storage": enable_structured_storage,
        "debug": debug,
        "display": display,
        "fast_mode": fast_mode,
    }

    storage_ctx = None
    if enable_structured_storage:
        storage_ctx = _init_structured_run_storage(storage_root=storage_root, run_name=run_name, run_config=run_config)
        print(f"structured storage run_id: {storage_ctx['run_id']}")
        print(f"structured storage dir: {storage_ctx['run_dir']}")
    # Initialize models and device
    device = select_device(device)
    yolo_model = YOLO(weights).to(device)
    # detect pose capability from resolved model task (more robust than filename checks)
    is_pose_model = getattr(yolo_model, "task", "") == "pose"

    # By default, save pose keypoints under this script's project root.
    if is_pose_model and save_kpts_dir is None:
        save_kpts_dir = str(PROJECT_ROOT / "keypoints")

    if save_kpts_dir is not None and not is_pose_model:
        print("warning: --save-kpts-dir was set, but the selected model is not a pose model; keypoints will not be saved.")

    # Virtual trainer setup
    references = load_reference_pose_library(reference_dir)
    capture_seed_references = (
        load_reference_pose_library(capture_seed_reference_dir)
        if capture_seed_reference_dir
        else {}
    )
    if label_thresholds_path is None:
        auto_threshold_path = Path(reference_dir) / "label_thresholds.json"
        if auto_threshold_path.exists():
            label_thresholds_path = str(auto_threshold_path)
    label_thresholds = _load_label_thresholds(label_thresholds_path)
    if label_thresholds:
        print(f"loaded per-label thresholds: {sorted(label_thresholds.keys())}")
    if record_reference:
        print(f"reference recording armed for technique: {record_reference}")
    if capture_seed_reference_dir:
        print(
            f"capture seed bank loaded from '{capture_seed_reference_dir}': "
            f"{sorted(list(capture_seed_references.keys()))}"
        )
    if trainer_enabled and is_pose_model:
        if not references and not record_reference:
            print(f"warning: no reference poses found in '{reference_dir}'. Use --record-reference to create one.")
        else:
            print(f"loaded reference techniques: {sorted(list(references.keys()))}")
        if evaluate_dir:
            if _normalize_key(target_technique) in references:
                eval_threshold = float(label_thresholds.get(_normalize_key(target_technique), trainer_score_threshold))
                eval_stats = evaluate_feedback_quality(
                    evaluate_dir=evaluate_dir,
                    references=references,
                    technique=target_technique,
                    score_threshold=eval_threshold,
                )
                if eval_stats is None:
                    print(f"no evaluation samples found for technique '{target_technique}' in {evaluate_dir}")
                else:
                    print(
                        "evaluation: "
                        f"thr={eval_threshold:.1f} "
                        f"n={eval_stats['total']} "
                        f"acc={eval_stats['accuracy']:.3f} "
                        f"prec={eval_stats['precision']:.3f} "
                        f"rec={eval_stats['recall']:.3f}"
                    )
            else:
                print(f"warning: cannot evaluate; no reference for technique '{target_technique}'")

    video_classifier = None
    if enable_video_classifier:
        if video_classifier_model in TorchVisionVideoClassifier.available_model_names():
            print("'fp16' is not supported for TorchVisionVideoClassifier. Setting fp16 to False.")
            print(
                "'labels' is not used for TorchVisionVideoClassifier. Ignoring the provided labels and using Kinetics-400 labels."
            )
            video_classifier = TorchVisionVideoClassifier(video_classifier_model, device=device)
        else:
            video_classifier = HuggingFaceVideoClassifier(
                labels, model_name=video_classifier_model, device=device, fp16=fp16
            )
    else:
        print("video classifier disabled: skipping action-label model initialization")

    # Initialize video capture
    source_str = str(source)
    webcam_index = int(source_str) if source_str.isdigit() else None

    if webcam_index is not None:
        cap = cv2.VideoCapture(webcam_index)
    elif source.startswith("http") and urlparse(source).hostname in {"www.youtube.com", "youtube.com", "youtu.be"}:
        try:
            source = get_best_youtube_url(source)
        except Exception as e:
            # some YouTube streams may have None resolutions; fall back to original URL
            print(f"warning: failed to select best YouTube stream ({e}), using original URL")
        cap = cv2.VideoCapture(source)
    elif Path(source).suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
        raise ValueError(
            f"Invalid source '{source}'. Supported sources are: webcam index (e.g. 0), "
            "YouTube URLs, or video files (.mp4, .avi, .mov, .mkv, .webm, .m4v)."
        )
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise ValueError(f"Failed to open source: {source}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = total_frames_raw if total_frames_raw > 0 else None
    if record_reference and auto_exit_after_reference and reference_search_max_frames > 0:
        total_frames = min(total_frames, reference_search_max_frames) if total_frames is not None else reference_search_max_frames

    # Initialize VideoWriter
    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            raise ValueError(f"Failed to open output writer: {out_path}")

    # Initialize track history (crops + optional keypoints)
    track_history: dict[int, list[np.ndarray]] = defaultdict(list)
    track_kpts_history: dict[int, list[np.ndarray]] = defaultdict(list) if is_pose_model else {}
    capture_kpts_history_len = num_video_sequence_samples
    if record_reference:
        capture_kpts_history_len = max(
            num_video_sequence_samples,
            num_video_sequence_samples * int(reference_capture_buffer_multiplier),
        )
    frame_counter = 0
    # prepare pose video writer if requested
    pose_writer = None
    if visualize_pose:
        if pose_output_path is None:
            # Default to the datasets folder under the project root.
            default_path = PROJECT_ROOT / "datasets" / "pose_video.mp4"
            default_path.parent.mkdir(parents=True, exist_ok=True)
            pose_output_path = str(default_path)
        else:
            pose_out_path = Path(pose_output_path)
            pose_out_path.parent.mkdir(parents=True, exist_ok=True)
            pose_output_path = str(pose_out_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        pose_writer = cv2.VideoWriter(pose_output_path, fourcc, fps, (frame_width, frame_height))
        if not pose_writer.isOpened():
            raise ValueError(f"Failed to open pose output writer: {pose_output_path}")

    track_ids_to_infer = []
    crops_to_infer: list[torch.Tensor] = []
    pred_labels: list = []
    pred_confs: list = []
    trainer_state: dict[int, dict[str, object]] = {}
    reference_saved_count = 0
    last_reference_saved_frame = -1
    best_reference_candidate: dict[str, object] | None = None
    track_box_history: dict[int, list[np.ndarray]] = defaultdict(list)
    primary_track_id: int | None = None
    primary_track_hold_until_frame = 0
    frame_progress = TQDM(total=total_frames, desc="processing frames", unit="frame")
    _exit_after_save = False

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_counter += 1

            if (
                record_reference
                and reference_saved_count == 0
                and auto_exit_after_reference
                and reference_search_max_frames > 0
                and frame_counter > reference_search_max_frames
            ):
                print(
                    f"reference search timed out after {reference_search_max_frames} frames for '{record_reference}'"
                )
                break

            # Run YOLO tracking (works for pose models too)

            results = yolo_model.track(frame, persist=True, classes=[0])  # Track only person class
            annotator = Annotator(frame, line_width=3, font_size=10, pil=False)
            pose_instances: list[np.ndarray] = []
            selected_pose_instances: list[np.ndarray] = []
            active_track_ids: list[int] = []
            current_track_boxes: dict[int, np.ndarray] = {}
            track_to_pose: dict[int, np.ndarray] = {}

            if results[0].boxes.is_track:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()

                # pull keypoints if provided by the model
                if is_pose_model:
                    pose_instances = extract_pose_instances(results[0], conf_thres=0.01)
                    if debug:
                        print(f"[debug] frame {frame_counter}: pose_instances={len(pose_instances)}")

                if frame_counter % skip_frame == 0:
                    crops_to_infer = []
                    track_ids_to_infer = []
                current_track_ids: list[int] = []

                for i, (box, track_id_raw) in enumerate(zip(boxes, track_ids)):
                    track_id = int(track_id_raw)
                    current_track_ids.append(track_id)
                    current_track_boxes[track_id] = np.asarray(box, dtype=np.float32)

                    if pose_instances and i < len(pose_instances):
                        track_to_pose[track_id] = pose_instances[i]

                    if frame_counter % skip_frame == 0:
                        crop = crop_and_pad(frame, box, crop_margin_percentage)
                        track_history[track_id].append(crop)
                        track_box_history[track_id].append(np.asarray(box, dtype=np.float32))
                        if pose_instances and i < len(pose_instances):
                            history = track_kpts_history.setdefault(track_id, [])
                            sanitized = _sanitize_kpt_entry(pose_instances[i])
                            if sanitized is not None:
                                history.append(sanitized)

                    if len(track_history[track_id]) > num_video_sequence_samples:
                        track_history[track_id].pop(0)
                    if len(track_box_history.get(track_id, [])) > num_video_sequence_samples:
                        track_box_history[track_id].pop(0)
                    if is_pose_model and len(track_kpts_history.get(track_id, [])) > capture_kpts_history_len:
                        track_kpts_history[track_id].pop(0)

                primary_track_id, selected_at_frame, track_scores = _select_primary_track(
                    track_ids=current_track_ids,
                    track_boxes=current_track_boxes,
                    track_kpts_history=track_kpts_history,
                    track_box_history=track_box_history,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    current_primary_track_id=primary_track_id,
                    current_frame=frame_counter,
                    hold_until_frame=primary_track_hold_until_frame,
                    primary_track_switch_margin=primary_track_switch_margin,
                    person_selection_mode=person_selection_mode,
                )
                if selected_at_frame != primary_track_hold_until_frame:
                    primary_track_hold_until_frame = selected_at_frame + primary_track_hold_frames

                active_track_ids = current_track_ids if person_selection_mode == "all" else ([primary_track_id] if primary_track_id is not None else [])
                active_boxes = [current_track_boxes[track_id] for track_id in active_track_ids if track_id in current_track_boxes]
                if person_selection_mode != "all" and primary_track_id is not None and primary_track_id in track_to_pose:
                    selected_pose_instances = [track_to_pose[primary_track_id]]
                else:
                    selected_pose_instances = [track_to_pose[track_id] for track_id in active_track_ids if track_id in track_to_pose]

                if person_selection_mode != "all" and primary_track_id is not None and draw_boxes and primary_track_id in current_track_boxes:
                    box = current_track_boxes[primary_track_id]
                    activity = track_scores.get(primary_track_id, 0.0)
                    annotator.box_label(box, f"fighter id {primary_track_id} | activity {activity:.2f}", color=(0, 255, 0))

                if frame_counter % skip_frame == 0 and primary_track_id is not None and primary_track_id in current_track_boxes:
                    if video_classifier is not None and len(track_history[primary_track_id]) >= num_video_sequence_samples:
                        start_time = time.time()
                        crops = video_classifier.preprocess_crops_for_video_cls(
                            track_history[primary_track_id][-num_video_sequence_samples:]
                        )
                        end_time = time.time()
                        preprocess_time = end_time - start_time
                        print(f"video cls preprocess time: {preprocess_time:.4f} seconds")
                        crops_to_infer.append(crops)
                        track_ids_to_infer.append(primary_track_id)

                    required_capture_frames = (
                        max(2, int(ref_stance_min_frames))
                        if reference_sequence_mode == "stance_cycle"
                        else num_video_sequence_samples
                    )
                    if trainer_enabled and is_pose_model and len(track_kpts_history.get(primary_track_id, [])) >= required_capture_frames:
                        stacked_seq = _safe_stack_kpt_sequence(track_kpts_history[primary_track_id])
                        if stacked_seq is not None and len(stacked_seq) >= required_capture_frames:
                            pose_seq = stacked_seq[-num_video_sequence_samples:]
                            capture_seq = pose_seq
                            if record_reference:
                                if reference_sequence_mode == "event_centered":
                                    event_window = _extract_event_centered_window(
                                        stacked_seq,
                                        window_len=num_video_sequence_samples,
                                    )
                                    if event_window is not None:
                                        capture_seq = event_window
                                elif reference_sequence_mode == "stance_cycle":
                                    stance_seq = _extract_stance_cycle_sequence(
                                        stacked_seq,
                                        start_threshold=float(ref_stance_start_threshold),
                                        end_threshold=float(ref_stance_end_threshold),
                                        peak_threshold=float(ref_stance_peak_threshold),
                                        min_frames=int(ref_stance_min_frames),
                                        hold_frames=int(ref_stance_hold_frames),
                                    )
                                    if stance_seq is not None:
                                        capture_seq = stance_seq

                            can_save_more = record_reference_max_saves == 0 or reference_saved_count < record_reference_max_saves
                            if record_reference and can_save_more:
                                if reference_capture_mode == "best_window":
                                    technique_key, _ = _split_reference_key(record_reference)
                                    energy = _wrist_motion_energy(capture_seq)
                                    closure = _wrist_return_closure(capture_seq)
                                    if energy >= ref_min_motion_energy and closure >= ref_min_return_closure:
                                        has_existing_refs = technique_key in references and bool(references[technique_key])
                                        passes_gate, gate_score = _passes_reference_score_gate(
                                            capture_seq, references, technique_key, ref_min_score_gate
                                        )
                                        passes_seed_gate, seed_score, has_seed_refs = _passes_reference_similarity_band(
                                            capture_seq,
                                            capture_seed_references,
                                            technique_key,
                                            capture_seed_min_score,
                                            capture_seed_max_score,
                                            bypass_if_missing=True,
                                        )
                                        if passes_gate and passes_seed_gate:
                                            # Prefer technique-match score when references exist, otherwise motion strength.
                                            if has_existing_refs:
                                                base_score = float(gate_score)
                                            elif has_seed_refs:
                                                base_score = float(seed_score)
                                            else:
                                                base_score = float(min(100.0, energy * 100.0))
                                            selection_score = base_score + 15.0 * float(closure)
                                            prev_best = float(best_reference_candidate["selection_score"]) if best_reference_candidate else -1.0
                                            if best_reference_candidate is None or selection_score > prev_best:
                                                best_reference_candidate = {
                                                    "pose_seq": capture_seq.copy(),
                                                    "frame": frame_counter,
                                                    "energy": float(energy),
                                                    "closure": float(closure),
                                                    "gate_score": float(gate_score),
                                                    "seed_score": float(seed_score),
                                                    "selection_score": float(selection_score),
                                                    "has_existing_refs": has_existing_refs,
                                                    "has_seed_refs": has_seed_refs,
                                                }
                                                if debug:
                                                    print(
                                                        f"[debug] best-window updated frame={frame_counter} "
                                                        f"selection={selection_score:.2f} motion={energy:.3f} "
                                                        f"closure={closure:.3f} gate={gate_score:.1f} seed={seed_score:.1f}"
                                                    )
                                else:
                                    append_indexed = record_reference_max_saves == 0 or record_reference_max_saves > 1
                                    saved_reference, last_reference_saved_frame = _try_record_reference_sequence(
                                        pose_seq=capture_seq,
                                        record_reference=record_reference,
                                        reference_dir=reference_dir,
                                        references=references,
                                        capture_seed_references=capture_seed_references,
                                        source=str(source),
                                        frame_counter=frame_counter,
                                        ref_min_motion_energy=ref_min_motion_energy,
                                        ref_min_return_closure=ref_min_return_closure,
                                        ref_min_score_gate=ref_min_score_gate,
                                        capture_seed_min_score=capture_seed_min_score,
                                        capture_seed_max_score=capture_seed_max_score,
                                        last_saved_frame=last_reference_saved_frame,
                                        reference_capture_cooldown_frames=reference_capture_cooldown_frames,
                                        append_indexed=append_indexed,
                                        enable_structured_storage=enable_structured_storage,
                                    )
                                    if saved_reference:
                                        reference_saved_count += 1
                                        if auto_exit_after_reference and record_reference_max_saves == 1:
                                            _exit_after_save = True

                            technique = _normalize_key(target_technique)
                            if technique in references:
                                best = _best_reference_match(
                                    user_sequence=pose_seq,
                                    reference_bank=references[technique],
                                    technique=technique,
                                )
                                if best is not None:
                                    best_angle, metrics = best
                                    score = float(metrics["score"])
                                    score_threshold = float(label_thresholds.get(technique, trainer_score_threshold))
                                    is_correct = score >= score_threshold
                                    feedback = generate_feedback(technique, pose_seq, score)
                                    ghost_pose = None
                                    matched_ref = references.get(technique, {}).get(best_angle)
                                    current_pose_abs = track_to_pose.get(primary_track_id)
                                    if not is_correct and matched_ref is not None and current_pose_abs is not None:
                                        ghost_pose, _ = _build_reference_overlay(
                                            user_sequence=pose_seq,
                                            reference_sequence=matched_ref,
                                            current_frame_kpts=current_pose_abs,
                                            technique=technique,
                                            use_mirror=bool(metrics.get("use_mirror", False)),
                                        )
                                    trainer_state[primary_track_id] = {
                                        "technique": technique,
                                        "reference_angle": best_angle,
                                        "score": score,
                                        "score_threshold": score_threshold,
                                        "is_correct": is_correct,
                                        "feedback": feedback,
                                        "ghost_pose": ghost_pose,
                                    }

                                    if enable_structured_storage and storage_ctx is not None:
                                        _append_metric_row(
                                            metrics_path=storage_ctx["metrics_path"],
                                            row={
                                                "run_id": storage_ctx["run_id"],
                                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                                "frame": frame_counter,
                                                "track_id": float(primary_track_id),
                                                "technique": technique,
                                                "reference_angle": best_angle,
                                                "score": f"{score:.4f}",
                                                "score_threshold": f"{score_threshold:.4f}",
                                                "is_correct": bool(is_correct),
                                                "cosine_similarity": f"{float(metrics['cosine_similarity']):.6f}",
                                                "dtw_distance": f"{float(metrics['dtw_distance']):.6f}",
                                                "angle_error": f"{float(metrics['angle_error']):.6f}",
                                                "mean_pose_distance": f"{float(metrics['mean_pose_distance']):.6f}",
                                                "use_mirror": bool(metrics["use_mirror"]),
                                                "feedback_1": feedback[0] if len(feedback) > 0 else "",
                                                "feedback_2": feedback[1] if len(feedback) > 1 else "",
                                                "source": str(source),
                                            },
                                        )

                if video_classifier is not None and crops_to_infer and (
                    not pred_labels
                    or frame_counter % video_cls_step == 0
                ):
                    crops_batch = torch.cat(crops_to_infer, dim=0)

                    start_inference_time = time.time()
                    output_batch = video_classifier(crops_batch)
                    end_inference_time = time.time()
                    inference_time = end_inference_time - start_inference_time
                    print(f"video cls inference time: {inference_time:.4f} seconds")

                    pred_labels, pred_confs = video_classifier.postprocess(output_batch)

                if video_classifier is not None and track_ids_to_infer and crops_to_infer:
                    if draw_boxes:
                        for track_id, pred_label, pred_conf in zip(track_ids_to_infer, pred_labels, pred_confs):
                            box = current_track_boxes.get(int(track_id))
                            if box is None:
                                continue
                            top2_preds = sorted(zip(pred_label, pred_conf), key=lambda x: x[1], reverse=True)
                            label_text = " | ".join([f"{label} ({conf:.2f})" for label, conf in top2_preds])
                            annotator.box_label(box, label_text, color=(0, 0, 255))

                # Draw trainer score/feedback near each tracked person.
                if trainer_enabled and is_pose_model:
                    for i, track_id in enumerate(active_track_ids):
                        box = current_track_boxes.get(track_id)
                        if box is None:
                            continue
                        state = trainer_state.get(int(track_id))
                        if not state:
                            continue
                        score = float(state["score"])
                        score_threshold = float(state.get("score_threshold", trainer_score_threshold))
                        is_correct = bool(state.get("is_correct", score >= score_threshold))
                        technique = str(state["technique"])
                        reference_angle = str(state.get("reference_angle", ""))
                        feedback = state["feedback"] if isinstance(state["feedback"], list) else []
                        display_technique = technique.replace("_", " ")
                        technique_line = _compact_text(
                            f"{display_technique} | ref {reference_angle}" if reference_angle else display_technique,
                            max_chars=48,
                        )
                        score_line = _compact_text(
                            f"score {score:.1f}/100  (target {score_threshold:.1f})",
                            max_chars=48,
                        )
                        tip_line = _compact_text(f"tip: {feedback[0]}", max_chars=48) if feedback else ""
                        panel_lines = [technique_line, score_line]
                        if tip_line:
                            panel_lines.append(tip_line)

                        if draw_boxes:
                            x1, y1, _, _ = map(int, box)
                            _draw_info_panel(
                                frame,
                                x=x1,
                                y=max(8, y1 - 78),
                                lines=panel_lines,
                                ok_state=is_correct,
                                font_scale=0.5,
                            )
                        else:
                            y_row = 10 + (i * 80)
                            left_panel_lines = [f"id {int(track_id)} | {technique_line}", score_line]
                            if tip_line:
                                left_panel_lines.append(tip_line)
                            _draw_info_panel(
                                frame,
                                x=10,
                                y=y_row,
                                lines=left_panel_lines,
                                ok_state=is_correct,
                                font_scale=0.5,
                            )

            # Show the image that Annotator actually draws on.
            display_frame = annotator.im

            # Overlay pose directly on the displayed frame.
            if overlay_pose and selected_pose_instances:
                for kp in selected_pose_instances:
                    draw_pose_cv2(display_frame, kp, conf_thres=0.01)

            if trainer_enabled and overlay_pose and is_pose_model:
                for track_id in active_track_ids:
                    state = trainer_state.get(int(track_id))
                    if not state or bool(state.get("is_correct", False)):
                        continue
                    ghost_pose = state.get("ghost_pose")
                    if ghost_pose is None:
                        continue
                    _draw_reference_ghost(display_frame, ghost_pose)

            if display:
                cv2.imshow("Video", display_frame)
            if output_path is not None:
                out.write(display_frame)

            if visualize_pose:
                pose_img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                if selected_pose_instances:
                    for kp in selected_pose_instances:
                        draw_pose_cv2(pose_img, kp, conf_thres=0.01)
                else:
                    cv2.putText(
                        pose_img,
                        "No pose detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                if display:
                    cv2.imshow("Pose", pose_img)
                if pose_writer is not None:
                    pose_writer.write(pose_img)

            frame_progress.update(1)

            # refresh windows and allow quit
            if display and (cv2.waitKey(1) & 0xFF == ord('q')):
                break
            if _exit_after_save:
                break
    finally:
        if (
            record_reference
            and reference_capture_mode == "best_window"
            and reference_saved_count == 0
            and best_reference_candidate is not None
        ):
            best_seq = best_reference_candidate["pose_seq"]
            if isinstance(best_seq, np.ndarray):
                append_indexed = record_reference_max_saves == 0 or record_reference_max_saves > 1
                ref_path = save_reference_pose(
                    reference_dir,
                    record_reference,
                    best_seq,
                    append_indexed=append_indexed,
                )
                technique_key, angle_key = _split_reference_key(record_reference)
                stored_angle = ref_path.stem if ref_path.parent.name == technique_key else (angle_key or "default")
                _put_reference(refs=references, technique=technique_key, angle=stored_angle, sequence=best_seq)

                best_frame = int(best_reference_candidate.get("frame", -1))
                best_motion = float(best_reference_candidate.get("energy", 0.0))
                best_closure = float(best_reference_candidate.get("closure", 0.0))
                gate_score = float(best_reference_candidate.get("gate_score", 0.0))
                seed_score = float(best_reference_candidate.get("seed_score", 0.0))
                selection_score = float(best_reference_candidate.get("selection_score", 0.0))
                has_existing_refs = bool(best_reference_candidate.get("has_existing_refs", False))
                has_seed_refs = bool(best_reference_candidate.get("has_seed_refs", False))
                gate_text = f"{gate_score:.1f}" if has_existing_refs else "bootstrap"
                seed_text = f", seed={seed_score:.1f}" if has_seed_refs else ""
                print(
                    f"saved best-window reference technique '{record_reference}' to: {ref_path} "
                    f"(selection={selection_score:.2f}, motion={best_motion:.3f}, closure={best_closure:.3f}, "
                    f"gate={gate_text}{seed_text}, frame={best_frame})"
                )
                if enable_structured_storage:
                    _write_reference_meta(
                        reference_path=ref_path,
                        technique=record_reference,
                        source=str(source),
                        num_frames=int(best_seq.shape[0]),
                    )
                reference_saved_count += 1
            else:
                print(f"warning: best-window candidate for '{record_reference}' was invalid and could not be saved")
        elif record_reference and reference_capture_mode == "best_window" and reference_saved_count == 0:
            print(f"no valid best-window candidate found for '{record_reference}'")

        if is_pose_model and save_kpts_dir is not None:
            save_keypoint_sequences(track_kpts_history, save_kpts_dir)
            print(f"saved keypoint sequences to: {Path(save_kpts_dir).resolve()}")

        if enable_structured_storage and storage_ctx is not None and is_pose_model:
            tracks_dir = storage_ctx["tracks_dir"]
            save_keypoint_sequences(track_kpts_history, str(tracks_dir))
            _write_track_summary(track_kpts_history, Path(storage_ctx["run_dir"]) / "tracks_summary.json")
            print(f"structured tracks saved to: {tracks_dir}")

        # always release resources
        cap.release()
        if output_path is not None:
            out.release()
        if pose_writer is not None:
            pose_writer.release()
        frame_progress.close()
        cv2.destroyAllWindows()
def parse_opt() -> argparse.Namespace:
    """Parse command line arguments for action recognition pipeline."""
    def _positive_int(value: str) -> int:
        ivalue = int(value)
        if ivalue < 1:
            raise argparse.ArgumentTypeError("value must be >= 1")
        return ivalue

    def _overlap_ratio(value: str) -> float:
        fvalue = float(value)
        if not (0.0 <= fvalue < 1.0):
            raise argparse.ArgumentTypeError("value must be in [0.0, 1.0)")
        return fvalue

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo26n-pose.pt",
        help="ultralytics detector model path (e.g. yolo26n-pose.pt for pose)",
    )
    parser.add_argument("--device", default="", help='cuda device, i.e. 0 or 0,1,2,3 or cpu/mps, "" for auto-detection')
    parser.add_argument(
        "--source",
        type=str,
        default="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        help="video file path, youtube URL, or webcam index (e.g. 0)",
    )
    parser.add_argument("--output-path", type=str, default="output_video.mp4", help="output video file path")
    parser.add_argument(
        "--crop-margin-percentage", type=int, default=10, help="percentage of margin to add around detected objects"
    )
    parser.add_argument(
        "--num-video-sequence-samples",
        type=_positive_int,
        default=8,
        help="number of video frames to use for classification",
    )
    parser.add_argument(
        "--skip-frame",
        type=_positive_int,
        default=1,
        help="number of frames to skip between detections",
    )
    parser.add_argument(
        "--video-cls-overlap-ratio",
        type=_overlap_ratio,
        default=0.25,
        help="overlap ratio between video sequences",
    )
    parser.add_argument("--fp16", action="store_true", help="use FP16 for inference")
    parser.add_argument(
        "--video-classifier-model", type=str, default="microsoft/xclip-base-patch32", help="video classifier model name"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=str,
        default=MARTIAL_ARTS_LABELS.copy(),
        help="labels for zero-shot video classification",
    )
    parser.add_argument(
        "--save-kpts-dir",
        type=str,
        default="keypoints",
        help="directory where per-track keypoint sequences will be saved (default: ./keypoints)",
    )
    parser.add_argument(
        "--visualize-pose",
        action="store_true",
        help="display a separate window showing only the pose skeleton on black",
    )
    parser.add_argument(
        "--pose-output-path",
        type=str,
        default=None,
        help="file to write the pose-only video (requires --visualize-pose). ``datasets/pose_video.mp4`` if omitted",
    )
    parser.add_argument(
        "--no-boxes",
        dest="draw_boxes",
        action="store_false",
        help="do not draw bounding boxes on the main video (useful for pose models)",
    )
    parser.add_argument(
        "--no-overlay-pose",
        dest="overlay_pose",
        action="store_false",
        help="do not overlay pose skeletons on the main video",
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="reference_poses",
        help=(
            "reference pose root. Supports legacy <technique>.npy and multi-angle "
            "<technique>/<angle>.npy layouts"
        ),
    )
    parser.add_argument(
        "--target-technique",
        type=str,
        default="jab",
        help="technique key to score against (e.g. jab, front_kick)",
    )
    parser.add_argument(
        "--disable-trainer",
        dest="trainer_enabled",
        action="store_false",
        help="disable live virtual-trainer scoring and feedback",
    )
    parser.add_argument(
        "--record-reference",
        type=str,
        default=None,
        help=(
            "save first complete tracked pose sequence as reference. "
            "Use <technique> for legacy flat save, or <technique>__<angle> for multi-angle save"
        ),
    )
    parser.add_argument(
        "--reference-capture-mode",
        type=str,
        default="first_valid",
        choices=["first_valid", "best_window", "event_centered"],
        help=(
            "reference capture strategy: 'first_valid' saves first accepted window, "
            "'best_window' scans candidate windows and saves the highest-scoring one, "
            "'event_centered' centers candidate windows around peak wrist extension"
        ),
    )
    parser.add_argument(
        "--reference-sequence-mode",
        type=str,
        default="fixed",
        choices=["fixed", "event_centered", "stance_cycle"],
        help=(
            "reference candidate extraction mode: 'fixed' uses latest fixed window, "
            "'event_centered' uses peak-extension centered window, "
            "'stance_cycle' extracts start/end from leave-stance to return-stance"
        ),
    )
    parser.add_argument(
        "--reference-capture-buffer-multiplier",
        type=int,
        default=3,
        help=(
            "rolling keypoint history multiplier used during reference capture; "
            "effective history length is num_video_sequence_samples * multiplier (default: 3)"
        ),
    )
    parser.add_argument(
        "--ref-stance-start-threshold",
        type=float,
        default=0.18,
        help="stance-cycle mode: pose-distance threshold to mark departure start (default: 0.18)",
    )
    parser.add_argument(
        "--ref-stance-end-threshold",
        type=float,
        default=0.12,
        help="stance-cycle mode: pose-distance threshold to mark return end (default: 0.12)",
    )
    parser.add_argument(
        "--ref-stance-peak-threshold",
        type=float,
        default=0.30,
        help="stance-cycle mode: minimum peak movement required (default: 0.30)",
    )
    parser.add_argument(
        "--ref-stance-min-frames",
        type=int,
        default=24,
        help="stance-cycle mode: minimum extracted sequence length (default: 24)",
    )
    parser.add_argument(
        "--ref-stance-hold-frames",
        type=int,
        default=4,
        help="stance-cycle mode: consecutive return frames required (default: 4)",
    )
    parser.add_argument(
        "--record-reference-max-saves",
        type=int,
        default=1,
        help="maximum accepted reference sequences to save from one video; 0 saves all accepted sequences",
    )
    parser.add_argument(
        "--reference-capture-cooldown-frames",
        type=int,
        default=24,
        help="minimum frame gap between saved reference sequences to avoid near-duplicate windows (default: 24)",
    )
    parser.add_argument(
        "--auto-exit-after-reference",
        dest="auto_exit_after_reference",
        action="store_true",
        help="exit immediately after the reference pose is saved (useful for batch pipeline runs)",
    )
    parser.add_argument(
        "--reference-search-max-frames",
        type=int,
        default=0,
        help="maximum frames to search for a valid reference before skipping the clip; 0 disables the limit (default: 0)",
    )
    parser.add_argument(
        "--ref-min-motion-energy",
        type=float,
        default=0.3,
        dest="ref_min_motion_energy",
        help=(
            "minimum wrist displacement (in shoulder-width units) required before a candidate "
            "reference sequence is accepted.  Rejects static / idle captures. (default: 0.3)"
        ),
    )
    parser.add_argument(
        "--ref-min-return-closure",
        type=float,
        default=0.15,
        dest="ref_min_return_closure",
        help=(
            "minimum wrist return-closure in [0,1] required for accepted reference capture. "
            "Higher values prefer full extension+retraction windows. (default: 0.15)"
        ),
    )
    parser.add_argument(
        "--ref-min-score-gate",
        type=float,
        default=75.0,
        dest="ref_min_score_gate",
        help=(
            "minimum similarity score (0-100) a candidate reference must achieve against "
            "already-stored references for this technique.  Bypassed when no reference exists yet. "
            "(default: 75.0)"
        ),
    )
    parser.add_argument(
        "--capture-seed-reference-dir",
        type=str,
        default=None,
        help=(
            "optional reference directory used only to seed reference capture. "
            "Useful for Golden Seed-derived .npy references when new captures should stay similar "
            "but not identical"
        ),
    )
    parser.add_argument(
        "--capture-seed-min-score",
        type=float,
        default=0.0,
        help=(
            "minimum similarity score a candidate must achieve against the capture seed bank "
            "before it can be saved (default: 0.0)"
        ),
    )
    parser.add_argument(
        "--capture-seed-max-score",
        type=float,
        default=100.0,
        help=(
            "maximum similarity score allowed against the capture seed bank. Lower this to reject "
            "near-identical copies of Golden Seed references (default: 100.0)"
        ),
    )
    parser.add_argument(
        "--person-selection-mode",
        type=str,
        default="most_motion",
        choices=["most_motion", "all"],
        help="select one active person to track/score, or process all detected people (default: most_motion)",
    )
    parser.add_argument(
        "--primary-track-hold-frames",
        type=int,
        default=15,
        help="frames to keep the current primary fighter before allowing a switch (default: 15)",
    )
    parser.add_argument(
        "--primary-track-switch-margin",
        type=float,
        default=1.15,
        help="new fighter must exceed current activity by this factor before switching (default: 1.15)",
    )
    parser.add_argument(
        "--trainer-score-threshold",
        type=float,
        default=70.0,
        help="score threshold used during offline evaluation to classify correct vs incorrect",
    )
    parser.add_argument(
        "--evaluate-dir",
        type=str,
        default=None,
        help="optional evaluation dataset root: <root>/<technique>/{correct,incorrect}/*.npy",
    )
    parser.add_argument(
        "--label-thresholds-path",
        type=str,
        default=None,
        help="optional JSON file with per-technique trainer thresholds",
    )
    parser.add_argument(
        "--disable-video-classifier",
        dest="enable_video_classifier",
        action="store_false",
        help="skip action classifier initialization/inference (pose and trainer only)",
    )
    parser.add_argument(
        "--storage-root",
        type=str,
        default="data",
        help="root directory for structured run artifacts",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="optional run name override (default: timestamp-based)",
    )
    parser.add_argument(
        "--disable-structured-storage",
        dest="enable_structured_storage",
        action="store_false",
        help="disable structured storage (runs/config/metrics/track summaries)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable verbose debug logging for frame-level internals",
    )
    parser.add_argument(
        "--no-display",
        dest="display",
        action="store_false",
        help="disable OpenCV display windows (faster/headless runs)",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help=(
            "apply speed-optimized preset: no display, no overlay/boxes, no video classifier, "
            "and higher frame skip"
        ),
    )
    return parser.parse_args()


def main(opt: argparse.Namespace) -> None:
    """Run the action recognition pipeline with parsed command line arguments."""
    # If run_name is not set, use the base name of the input video (without extension) as default
    if getattr(opt, 'run_name', None) in (None, ''):
        source = getattr(opt, 'source', None)
        if source and not source.isdigit() and not source.startswith('http'):
            from pathlib import Path
            opt.run_name = Path(source).stem
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)