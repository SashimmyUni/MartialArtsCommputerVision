"""Check the debounce/merge logic in technique_events.py.

A frame-level classifier tick fires far more often than a fighter actually
throws a technique, so ``segment_events`` must merge consecutive same-label
ticks into one event, bridge small gaps (a missed tick), split on a genuine
label change, and drop events that are too short to trust. This is the
correctness gate for "most-used technique" counts: get this wrong and one jab
gets counted five times, or a real leg kick gets bridged into an unrelated
cross.

Needs only the stdlib. Run from anywhere:

    python test_technique_events.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from technique_events import FramePrediction, segment_events  # noqa: E402


def _pred(frame: int, technique: str | None, confidence: float = 0.8, dtw_score: float | None = None) -> FramePrediction:
    return FramePrediction(
        frame=frame,
        time_s=frame / 30.0,
        technique=technique,
        family="punch",
        confidence=confidence,
        detection_method="hybrid" if dtw_score is not None else "classifier_only",
        dtw_score=dtw_score,
    )


def test_single_contiguous_run_merges_to_one_event() -> None:
    preds = [_pred(f, "jab", confidence=c) for f, c in zip([10, 11, 12, 13], [0.6, 0.7, 0.9, 0.8])]
    events = segment_events(preds, fighter_id="fighter_1")
    assert len(events) == 1, f"expected 1 event, got {len(events)}"
    ev = events[0]
    assert ev.technique == "jab"
    assert ev.start_frame == 10 and ev.end_frame == 13
    assert ev.frame_count == 4
    assert abs(ev.confidence - 0.75) < 1e-9, ev.confidence
    print("OK: contiguous run merges into one event")


def test_small_gap_is_bridged() -> None:
    # Frame 16 is missing (a skipped/failed tick) but the label resumes -- still one event.
    preds = [_pred(10, "cross"), _pred(13, "cross"), _pred(19, "cross")]
    events = segment_events(preds, fighter_id="fighter_1", max_gap_frames=6)
    assert len(events) == 1, f"expected bridged single event, got {len(events)}"
    assert events[0].start_frame == 10 and events[0].end_frame == 19
    print("OK: small gap bridged into one event")


def test_large_gap_splits_into_two_events() -> None:
    preds = [_pred(10, "hook"), _pred(11, "hook"), _pred(50, "hook"), _pred(51, "hook")]
    events = segment_events(preds, fighter_id="fighter_1", max_gap_frames=2)
    assert len(events) == 2, f"expected 2 events, got {len(events)}"
    assert events[0].end_frame == 11
    assert events[1].start_frame == 50
    print("OK: gap larger than tolerance splits into two events")


def test_label_change_splits_events_at_boundary() -> None:
    preds = [_pred(10, "jab"), _pred(11, "jab"), _pred(12, "cross"), _pred(13, "cross")]
    events = segment_events(preds, fighter_id="fighter_1")
    assert [e.technique for e in events] == ["jab", "cross"]
    assert events[0].end_frame == 11
    assert events[1].start_frame == 12
    print("OK: differing label starts a new event at the exact boundary")


def test_min_duration_drops_short_events() -> None:
    preds = [_pred(10, "uppercut")]  # a single flickering tick
    events = segment_events(preds, fighter_id="fighter_1", min_duration_ticks=2)
    assert events == [], f"expected the lone tick dropped, got {events}"

    kept = segment_events(preds, fighter_id="fighter_1", min_duration_ticks=1)
    assert len(kept) == 1, "min_duration_ticks=1 must keep a single-tick event"
    print("OK: events shorter than min_duration_ticks are dropped")


def test_none_technique_and_empty_input_are_ignored() -> None:
    assert segment_events([], fighter_id="fighter_1") == []
    preds = [_pred(10, None), _pred(11, None)]
    assert segment_events(preds, fighter_id="fighter_1") == []
    print("OK: empty input and unclassified ticks produce no events")


def test_dtw_score_averaged_only_over_confirmed_ticks() -> None:
    preds = [
        _pred(10, "front_kick", dtw_score=60.0),
        _pred(11, "front_kick", dtw_score=None),  # classifier-only tick, no DTW confirmation
        _pred(12, "front_kick", dtw_score=80.0),
    ]
    events = segment_events(preds, fighter_id="fighter_2")
    assert len(events) == 1
    assert abs(events[0].dtw_score - 70.0) < 1e-9, events[0].dtw_score
    assert events[0].fighter_id == "fighter_2"
    print("OK: dtw_score averages only over ticks that had one")


def test_unsorted_input_is_sorted_by_frame() -> None:
    preds = [_pred(12, "jab"), _pred(10, "jab"), _pred(11, "jab")]
    events = segment_events(preds, fighter_id="fighter_1")
    assert len(events) == 1
    assert events[0].start_frame == 10 and events[0].end_frame == 12
    print("OK: out-of-order predictions are sorted before segmenting")


if __name__ == "__main__":
    test_single_contiguous_run_merges_to_one_event()
    test_small_gap_is_bridged()
    test_large_gap_splits_into_two_events()
    test_label_change_splits_events_at_boundary()
    test_min_duration_drops_short_events()
    test_none_technique_and_empty_input_are_ignored()
    test_dtw_score_averaged_only_over_confirmed_ticks()
    test_unsorted_input_is_sorted_by_frame()
    print("\nOK: all technique_events tests passed")
