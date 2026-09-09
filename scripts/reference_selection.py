"""Stage 2: pick reference windows by replaying a cached detection table.

Everything the capture loop does *after* the pose model is pure numpy over the
per-frame detections: build rolling per-track histories, pick the primary
fighter, cut a candidate window, score it against the gates. None of it needs a
GPU, a video file, or a network connection — it only needed those because it
was welded to the decode-and-infer loop.

Replaying it from ``cache/tracks/`` instead buys two things:

- One video pass can yield several reference examples, because the whole run is
  in memory at once instead of one best candidate being kept and the process
  exiting (``action_recognition.run()`` saves its ``best_window`` pick in a
  ``finally`` block, so N examples meant N full passes).
- Re-tuning a gate costs seconds instead of a full re-download and
  re-inference. ``docs/HOWTO.md`` §9.1 calls reference quality the top-priority
  problem; this is what makes iterating on it affordable.

The replay reuses ``action_recognition``'s own selection helpers rather than
reimplementing them, so there is one definition of what a valid reference is.
What this module must get right is the *state* those helpers see on each frame,
which is what ``iter_candidates`` reproduces and ``test_capture_equivalence.py``
pins down.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import action_recognition as ar  # noqa: E402
import reference_cache as rc  # noqa: E402


@dataclass
class SelectionConfig:
    """Everything that decides which window gets picked.

    Deliberately disjoint from ``reference_cache.extraction_signature``: none of
    these appear in the cache key, which is why changing any of them replays the
    same cached tracks instead of re-running the model.
    """

    num_video_sequence_samples: int = 20
    reference_sequence_mode: str = "stance_cycle"
    reference_capture_buffer_multiplier: int = 3
    skip_frame: int = 1
    person_selection_mode: str = "most_motion"
    primary_track_hold_frames: int = 15
    primary_track_switch_margin: float = 1.15

    ref_min_motion_energy: float = 0.03
    ref_min_return_closure: float = 0.22
    ref_min_score_gate: float = 0.0
    ref_stance_start_threshold: float = 0.17
    ref_stance_end_threshold: float = 0.12
    ref_stance_peak_threshold: float = 0.30
    ref_stance_min_frames: int = 20
    ref_stance_hold_frames: int = 4

    capture_seed_min_score: float = 0.0
    capture_seed_max_score: float = 100.0

    search_max_frames: int = 1800
    segment_start_frame: int = 0
    segment_end_frame: int = 0  # 0 = to the end of what was extracted

    @property
    def capture_kpts_history_len(self) -> int:
        """Rolling keypoint buffer length, as ``run()`` computes it when capturing."""
        return max(
            self.num_video_sequence_samples,
            self.num_video_sequence_samples * int(self.reference_capture_buffer_multiplier),
        )

    @property
    def required_capture_frames(self) -> int:
        if self.reference_sequence_mode == "stance_cycle":
            return max(2, int(self.ref_stance_min_frames))
        return self.num_video_sequence_samples

    @classmethod
    def from_profile(cls, profile: dict, **overrides) -> "SelectionConfig":
        """Build from a ``TECHNIQUE_CAPTURE_PROFILES`` entry.

        Keeps ``run_reference_collection_batch.py``'s per-technique profiles as
        the single source of truth for these values instead of restating them.
        """
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in profile.items() if k in known}
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kwargs)


# eq=False: the generated __eq__ would compare pose_seq arrays elementwise, so
# any list operation that tests equality (list.remove, `in`) raises "truth value
# of an array is ambiguous". Candidates are unique objects; identity is the
# right comparison.
@dataclass(eq=False)
class Candidate:
    """One gate-passing window, with where in the video it came from."""

    pose_seq: np.ndarray
    end_frame: int          # frame the rolling buffer ended on
    start_frame: int        # first source frame contributing to the window
    last_frame: int         # last source frame contributing to the window
    energy: float
    closure: float
    gate_score: float = 0.0
    seed_score: float = 0.0
    selection_score: float = 0.0
    has_existing_refs: bool = False
    has_seed_refs: bool = False

    @property
    def span(self) -> tuple[int, int]:
        return self.start_frame, self.last_frame

    def center(self) -> float:
        return (self.start_frame + self.last_frame) / 2.0


@dataclass
class _TrackState:
    kpts: list[np.ndarray] = field(default_factory=list)
    kpt_frames: list[int] = field(default_factory=list)
    boxes: list[np.ndarray] = field(default_factory=list)


def iter_candidates(cache: rc.TrackCache, config: SelectionConfig) -> list[Candidate]:
    """Replay the capture loop over a cached table and collect every window it would consider.

    This mirrors ``action_recognition.run()``'s frame loop exactly, minus decode,
    inference and drawing. The order of operations matters more than it looks:

    - Histories are appended to and trimmed **inside** the per-detection loop,
      and only for tracks seen on that frame, so a track that disappears keeps a
      stale buffer that is still scored later.
    - Box and keypoint buffers have *different* caps
      (``num_video_sequence_samples`` vs ``capture_kpts_history_len``), and
      ``_track_activity_score`` reads both.
    - ``_select_primary_track`` is pure but hysteretic: the primary id and the
      hold-until frame thread through the whole run.
    - ``stance_cycle`` windows anchor on the *oldest* frame in the buffer, so
      buffer alignment decides the result.

    Unlike the live path, which keeps only a running best, every candidate is
    kept — that is what makes top-K from a single pass possible.
    """
    if config.person_selection_mode == "all":
        # run() never selects a primary track in this mode, so nothing is captured.
        return []

    tracks: dict[int, _TrackState] = {}
    primary_track_id: int | None = None
    hold_until_frame = 0
    candidates: list[Candidate] = []

    last_frame = cache.frames_read
    if config.search_max_frames > 0:
        last_frame = min(last_frame, config.search_max_frames)
    first_frame = max(1, int(config.segment_start_frame) or 1)
    if config.segment_end_frame > 0:
        last_frame = min(last_frame, int(config.segment_end_frame))

    box_cap = config.num_video_sequence_samples
    kpt_cap = config.capture_kpts_history_len
    required = config.required_capture_frames

    for frame in range(first_frame, last_frame + 1):
        track_ids, boxes = cache.boxes_for_frame(frame)
        if track_ids.size == 0:
            # run() skips the whole block when nothing is tracked on this frame.
            continue
        poses = cache.poses_for_frame(frame)

        current_track_boxes: dict[int, np.ndarray] = {}
        current_track_ids: list[int] = []
        append_this_frame = (frame % config.skip_frame) == 0

        for i, (tid_raw, box) in enumerate(zip(track_ids, boxes)):
            tid = int(tid_raw)
            current_track_ids.append(tid)
            current_track_boxes[tid] = np.asarray(box, dtype=np.float32)
            state = tracks.setdefault(tid, _TrackState())

            if append_this_frame:
                state.boxes.append(np.asarray(box, dtype=np.float32))
                # Positional pairing, matching run(): pose i belongs to box i, and
                # a pose dropped by extract_pose_instances shifts every later pair.
                if poses.shape[0] and i < poses.shape[0]:
                    sanitized = ar._sanitize_kpt_entry(poses[i])
                    if sanitized is not None:
                        state.kpts.append(sanitized)
                        state.kpt_frames.append(frame)

            if len(state.boxes) > box_cap:
                state.boxes.pop(0)
            if len(state.kpts) > kpt_cap:
                state.kpts.pop(0)
                state.kpt_frames.pop(0)

        primary_track_id, selected_at_frame, _scores, _stacked = ar._select_primary_track(
            track_ids=current_track_ids,
            track_boxes=current_track_boxes,
            track_kpts_history={tid: tracks[tid].kpts for tid in tracks},
            track_box_history={tid: tracks[tid].boxes for tid in tracks},
            frame_width=cache.frame_width,
            frame_height=cache.frame_height,
            current_primary_track_id=primary_track_id,
            current_frame=frame,
            hold_until_frame=hold_until_frame,
            primary_track_switch_margin=config.primary_track_switch_margin,
            person_selection_mode=config.person_selection_mode,
        )
        if selected_at_frame != hold_until_frame:
            hold_until_frame = selected_at_frame + config.primary_track_hold_frames

        if not append_this_frame or primary_track_id is None or primary_track_id not in current_track_boxes:
            continue

        state = tracks.get(primary_track_id)
        if state is None or len(state.kpts) < required:
            continue

        stacked = ar._safe_stack_kpt_sequence(state.kpts)
        if stacked is None or len(stacked) < required:
            continue
        if len(stacked) != len(state.kpt_frames):
            # _safe_stack_kpt_sequence drops entries whose keypoint shape differs
            # from the majority. Extraction refuses to cache mixed shapes, so this
            # should not happen — but if it ever does, the buffer-index to
            # frame-number mapping is broken and the window's span would be a
            # guess. Skip rather than report a span we cannot stand behind.
            continue

        buffer_frames = state.kpt_frames
        span = _window_span(stacked, config)
        if span is None:
            continue
        lo, hi = span
        capture_seq = stacked[lo:hi]
        if capture_seq.shape[0] == 0:
            continue

        energy = float(ar._wrist_motion_energy(capture_seq))
        closure = float(ar._wrist_return_closure(capture_seq))
        if energy < config.ref_min_motion_energy or closure < config.ref_min_return_closure:
            continue

        candidates.append(
            Candidate(
                pose_seq=capture_seq.copy(),
                end_frame=frame,
                start_frame=int(buffer_frames[lo]),
                last_frame=int(buffer_frames[hi - 1]),
                energy=energy,
                closure=closure,
            )
        )

    return candidates


def _window_span(stacked: np.ndarray, config: SelectionConfig) -> tuple[int, int] | None:
    """Half-open buffer span of the candidate window, per reference_sequence_mode.

    Falls back to the trailing ``num_video_sequence_samples`` frames exactly as
    ``run()`` does when the mode-specific extractor declines.
    """
    total = int(stacked.shape[0])
    fallback = (max(0, total - config.num_video_sequence_samples), total)

    if config.reference_sequence_mode == "event_centered":
        span = ar._event_centered_span(stacked, window_len=config.num_video_sequence_samples)
        return span if span is not None else fallback
    if config.reference_sequence_mode == "stance_cycle":
        span = ar._stance_cycle_span(
            stacked,
            start_threshold=float(config.ref_stance_start_threshold),
            end_threshold=float(config.ref_stance_end_threshold),
            peak_threshold=float(config.ref_stance_peak_threshold),
            min_frames=int(config.ref_stance_min_frames),
            hold_frames=int(config.ref_stance_hold_frames),
        )
        return span if span is not None else fallback
    return fallback


def score_candidate(
    candidate: Candidate,
    references: dict[str, dict[str, np.ndarray]],
    seed_references: dict[str, dict[str, np.ndarray]],
    technique_key: str,
    config: SelectionConfig,
) -> bool:
    """Apply the reference and seed-band gates, filling in the selection score.

    Reproduces ``run()``'s ranking verbatim, including the detail that with
    ``--ref-min-score-gate 0`` the gate never *rejects* — it only supplies
    ``gate_score``, which becomes the ranking base whenever the technique
    already has references. Returns False when the candidate is rejected.
    """
    passes_gate, gate_score = ar._passes_reference_score_gate(
        candidate.pose_seq, references, technique_key, config.ref_min_score_gate
    )
    passes_seed, seed_score, has_seed_refs = ar._passes_reference_similarity_band(
        candidate.pose_seq,
        seed_references,
        technique_key,
        config.capture_seed_min_score,
        config.capture_seed_max_score,
        bypass_if_missing=True,
    )
    if not (passes_gate and passes_seed):
        return False

    tkey = ar._normalize_key(technique_key)
    has_existing_refs = tkey in references and bool(references[tkey])
    if has_existing_refs:
        base_score = float(gate_score)
    elif has_seed_refs:
        base_score = float(seed_score)
    else:
        base_score = float(min(100.0, candidate.energy * 100.0))

    candidate.gate_score = float(gate_score)
    candidate.seed_score = float(seed_score)
    candidate.has_existing_refs = bool(has_existing_refs)
    candidate.has_seed_refs = bool(has_seed_refs)
    candidate.selection_score = base_score + 15.0 * float(candidate.closure)
    return True


def _windows_overlap(a: Candidate, b: Candidate, min_gap_frames: int) -> bool:
    """Do two candidates cover the same moment, or sit too close together?"""
    a_lo, a_hi = a.span
    b_lo, b_hi = b.span
    if a_lo <= b_hi and b_lo <= a_hi:
        return True
    return abs(a.center() - b.center()) < float(min_gap_frames)


def select_top_k(
    candidates: list[Candidate],
    k: int,
    *,
    references: dict[str, dict[str, np.ndarray]],
    seed_references: dict[str, dict[str, np.ndarray]],
    technique_key: str,
    config: SelectionConfig,
    min_gap_frames: int = 24,
    max_self_similarity: float = 95.0,
    rescore_after_each: bool = True,
) -> list[Candidate]:
    """Greedily take the best ``k`` windows that are not the same moment twice.

    Three things keep the K windows from collapsing onto one action:

    1. Their source frame spans must be disjoint, and their centres at least
       ``min_gap_frames`` apart — the same idea as the live path's
       ``reference_capture_cooldown_frames``, reusing that value rather than
       introducing a second knob.
    2. An accepted window must not score above ``max_self_similarity`` against
       one already taken. Overlap alone does not catch a repeated jab thrown
       identically eight frames later, which is exactly what a tutorial video is
       full of. Set to 0 to disable.
    3. With ``rescore_after_each``, an accepted window joins the in-memory bank
       and the remaining candidates are re-ranked against it. This is what the
       old flow did implicitly: each example ran in its own subprocess and
       reloaded the library from disk, so example 02 was already ranked against
       example 01.
    """
    if k <= 0:
        return []

    tkey = ar._normalize_key(technique_key)
    # Rescoring adds accepted windows to the bank, so work on a copy: the caller's
    # library must end up holding only what was actually saved to disk, or the
    # next row would be ranked against provisional windows this one discarded.
    working_refs = dict(references)
    working_refs[tkey] = dict(working_refs.get(tkey, {}))

    scored = [
        c for c in candidates
        if score_candidate(c, working_refs, seed_references, technique_key, config)
    ]
    accepted: list[Candidate] = []

    while scored and len(accepted) < k:
        # Ties break toward the earlier window, so a rerun picks the same one.
        scored.sort(key=lambda c: (-c.selection_score, c.end_frame))
        pick = None
        for candidate in scored:
            if any(_windows_overlap(candidate, taken, min_gap_frames) for taken in accepted):
                continue
            if max_self_similarity > 0 and accepted:
                bank = {f"_taken_{i}": taken.pose_seq for i, taken in enumerate(accepted)}
                best = ar._best_reference_match(candidate.pose_seq, bank, tkey)
                if best is not None and float(best[1]["score"]) >= max_self_similarity:
                    continue
            pick = candidate
            break

        if pick is None:
            break

        accepted.append(pick)
        scored.remove(pick)

        if rescore_after_each and scored:
            ar._put_reference(
                refs=working_refs,
                technique=tkey,
                angle=f"pending_{len(accepted):02d}",
                sequence=pick.pose_seq,
            )
            scored = [
                c for c in scored
                if score_candidate(c, working_refs, seed_references, technique_key, config)
            ]

    return accepted
