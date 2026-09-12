"""Collapse per-frame technique predictions into discrete technique events.

A frame-level classifier tick fires many times a second; one thrown jab shows
up as several consecutive identical predictions. This module runs a
minimum-duration + small-gap-bridging pass over that per-fighter prediction
stream so usage statistics count *events* (technique instances thrown), not
classifier ticks. Deliberately dependency-free (stdlib only) so it can be
imported by both ``fight_analysis.py`` and ``scripts/analyze_fight_metrics.py``
without pulling in torch/numpy.

``max_gap_frames`` is expressed in raw video frame numbers (bridges a missed
classifier tick); ``min_duration_ticks`` counts predictions in the merged
group. Callers whose predictions arrive every ``video_cls_step`` frames (not
every frame) should scale the gap accordingly, e.g.
``max_gap_frames = video_cls_step * max_gap_ticks``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FramePrediction:
    """One fighter's fused technique guess at one classifier tick."""

    frame: int
    time_s: float
    technique: str | None
    family: str | None
    confidence: float
    detection_method: str
    dtw_score: float | None = None


@dataclass
class TechniqueEvent:
    """One collapsed technique occurrence (a single thrown technique)."""

    fighter_id: str
    technique: str
    family: str | None
    detection_method: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    confidence: float
    dtw_score: float | None
    frame_count: int

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time_s - self.start_time_s)


def segment_events(
    predictions: list[FramePrediction],
    fighter_id: str,
    min_duration_ticks: int = 1,
    max_gap_frames: int = 1,
) -> list[TechniqueEvent]:
    """Collapse a time-ordered per-fighter prediction stream into events.

    Consecutive predictions sharing the same technique label are merged into
    one event. A frame gap of up to ``max_gap_frames`` between two same-label
    predictions is bridged (treated as a missed/skipped tick, not a new
    event); a larger gap, or a differing label, starts a new event. Events
    with fewer than ``min_duration_ticks`` merged predictions are dropped as
    classifier noise. Predictions with ``technique=None`` are ignored (no
    confident guess that tick).
    """
    ordered = sorted((p for p in predictions if p.technique), key=lambda p: p.frame)
    events: list[TechniqueEvent] = []
    if not ordered:
        return events

    def _flush(group: list[FramePrediction]) -> None:
        if len(group) < min_duration_ticks:
            return
        confidences = [p.confidence for p in group]
        dtw_scores = [p.dtw_score for p in group if p.dtw_score is not None]
        events.append(
            TechniqueEvent(
                fighter_id=fighter_id,
                technique=str(group[0].technique),
                family=group[0].family,
                detection_method=group[0].detection_method,
                start_frame=group[0].frame,
                end_frame=group[-1].frame,
                start_time_s=group[0].time_s,
                end_time_s=group[-1].time_s,
                confidence=sum(confidences) / len(confidences),
                dtw_score=(sum(dtw_scores) / len(dtw_scores)) if dtw_scores else None,
                frame_count=len(group),
            )
        )

    current_group = [ordered[0]]
    for prev, pred in zip(ordered, ordered[1:]):
        same_label = pred.technique == current_group[-1].technique
        gap = pred.frame - prev.frame
        if same_label and gap <= max_gap_frames + 1:
            current_group.append(pred)
            continue
        _flush(current_group)
        current_group = [pred]
    _flush(current_group)
    return events
