"""Verify the cached-track replay picks the same reference window as the live capture loop.

Reference capture used to be welded to the decode-and-infer loop in
``action_recognition.run()``: one process per saved ``.npy``, each re-decoding
and re-inferring a video to keep a single best window. ``reference_selection``
replays the same decisions from a cached detection table instead, which is what
lets one video pass yield several examples and makes re-tuning a gate cost
seconds rather than a full re-run.

That is only safe if the replay is faithful. This file pins it down the same way
``test_scoring_equivalence.py`` pins the scoring core: a copy of the original
selection block, transcribed from ``run()``'s frame loop with decode, inference
and drawing removed, is used as ground truth, and the replay must agree with it
exactly.

The parts most likely to drift, and the reason each is exercised below:

- Box and keypoint buffers have **different** caps, and both feed
  ``_track_activity_score``.
- Buffers are appended and trimmed **inside** the per-detection loop and only
  for tracks seen on that frame, so a track that vanishes keeps a stale buffer.
- ``_select_primary_track`` is hysteretic — the primary id and hold-until frame
  thread through the whole run.
- ``stance_cycle`` anchors on the **oldest** frame in the buffer, so a
  one-frame buffer misalignment changes the window.
- ``extract_pose_instances`` can return fewer poses than boxes, and ``run()``
  still pairs them positionally.

Needs numpy and the project's own modules; no GPU, no video, no network.

    python test_capture_equivalence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "scripts"))

import action_recognition as ar  # noqa: E402
import reference_cache as rc  # noqa: E402
import reference_selection as rs  # noqa: E402

TOL = 1e-5


# ---------------------------------------------------------------------------
# Pinned reference implementation: action_recognition.run()'s capture path,
# transcribed with decode/inference/drawing stripped out. Ground truth only —
# never used by the application.
# ---------------------------------------------------------------------------

def legacy_best_window(
    cache: rc.TrackCache,
    config: rs.SelectionConfig,
    references: dict,
    seed_references: dict,
    technique_key: str,
) -> dict | None:
    """The single best candidate, exactly as the live ``best_window`` path finds it."""
    track_kpts_history: dict[int, list] = {}
    track_box_history: dict[int, list] = {}
    primary_track_id = None
    primary_track_hold_until_frame = 0
    best_reference_candidate = None

    num_video_sequence_samples = config.num_video_sequence_samples
    capture_kpts_history_len = max(
        num_video_sequence_samples,
        num_video_sequence_samples * int(config.reference_capture_buffer_multiplier),
    )
    skip_frame = config.skip_frame

    total = cache.frames_read
    if config.search_max_frames > 0:
        total = min(total, config.search_max_frames)

    for frame_counter in range(1, total + 1):
        track_ids, boxes = cache.boxes_for_frame(frame_counter)
        if track_ids.size == 0:
            continue
        pose_instances = list(cache.poses_for_frame(frame_counter))

        current_track_ids: list[int] = []
        current_track_boxes: dict[int, np.ndarray] = {}

        for i, (box, track_id_raw) in enumerate(zip(boxes, track_ids)):
            track_id = int(track_id_raw)
            current_track_ids.append(track_id)
            current_track_boxes[track_id] = np.asarray(box, dtype=np.float32)

            if frame_counter % skip_frame == 0:
                track_box_history.setdefault(track_id, []).append(np.asarray(box, dtype=np.float32))
                if pose_instances and i < len(pose_instances):
                    history = track_kpts_history.setdefault(track_id, [])
                    sanitized = ar._sanitize_kpt_entry(pose_instances[i])
                    if sanitized is not None:
                        history.append(sanitized)

            if len(track_box_history.get(track_id, [])) > num_video_sequence_samples:
                track_box_history[track_id].pop(0)
            if len(track_kpts_history.get(track_id, [])) > capture_kpts_history_len:
                track_kpts_history[track_id].pop(0)

        primary_track_id, selected_at_frame, _scores, stacked_kpts_by_track = ar._select_primary_track(
            track_ids=current_track_ids,
            track_boxes=current_track_boxes,
            track_kpts_history=track_kpts_history,
            track_box_history=track_box_history,
            frame_width=cache.frame_width,
            frame_height=cache.frame_height,
            current_primary_track_id=primary_track_id,
            current_frame=frame_counter,
            hold_until_frame=primary_track_hold_until_frame,
            primary_track_switch_margin=config.primary_track_switch_margin,
            person_selection_mode=config.person_selection_mode,
        )
        if selected_at_frame != primary_track_hold_until_frame:
            primary_track_hold_until_frame = selected_at_frame + config.primary_track_hold_frames

        if not (frame_counter % skip_frame == 0 and primary_track_id is not None
                and primary_track_id in current_track_boxes):
            continue

        required_capture_frames = (
            max(2, int(config.ref_stance_min_frames))
            if config.reference_sequence_mode == "stance_cycle"
            else num_video_sequence_samples
        )
        if len(track_kpts_history.get(primary_track_id, [])) < required_capture_frames:
            continue

        stacked_seq = stacked_kpts_by_track.get(primary_track_id)
        if stacked_seq is None:
            stacked_seq = ar._safe_stack_kpt_sequence(track_kpts_history[primary_track_id])
        if stacked_seq is None or len(stacked_seq) < required_capture_frames:
            continue

        pose_seq = stacked_seq[-num_video_sequence_samples:]
        capture_seq = pose_seq
        if config.reference_sequence_mode == "event_centered":
            event_window = ar._extract_event_centered_window(
                stacked_seq, window_len=num_video_sequence_samples
            )
            if event_window is not None:
                capture_seq = event_window
        elif config.reference_sequence_mode == "stance_cycle":
            stance_seq = ar._extract_stance_cycle_sequence(
                stacked_seq,
                start_threshold=float(config.ref_stance_start_threshold),
                end_threshold=float(config.ref_stance_end_threshold),
                peak_threshold=float(config.ref_stance_peak_threshold),
                min_frames=int(config.ref_stance_min_frames),
                hold_frames=int(config.ref_stance_hold_frames),
            )
            if stance_seq is not None:
                capture_seq = stance_seq

        energy = ar._wrist_motion_energy(capture_seq)
        closure = ar._wrist_return_closure(capture_seq)
        if not (energy >= config.ref_min_motion_energy and closure >= config.ref_min_return_closure):
            continue

        has_existing_refs = (
            ar._normalize_key(technique_key) in references
            and bool(references[ar._normalize_key(technique_key)])
        )
        passes_gate, gate_score = ar._passes_reference_score_gate(
            capture_seq, references, technique_key, config.ref_min_score_gate
        )
        passes_seed_gate, seed_score, has_seed_refs = ar._passes_reference_similarity_band(
            capture_seq,
            seed_references,
            technique_key,
            config.capture_seed_min_score,
            config.capture_seed_max_score,
            bypass_if_missing=True,
        )
        if not (passes_gate and passes_seed_gate):
            continue

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
            }

    return best_reference_candidate


# ---------------------------------------------------------------------------
# Synthetic detection tables
# ---------------------------------------------------------------------------

def _committed_pose_pool() -> np.ndarray:
    """Real keypoint frames from the committed fixtures, as a (T,17,3) pool."""
    kp_dir = SCRIPT_DIR / "keypoints"
    frames = []
    for path in sorted(kp_dir.glob("track_*.npy")):
        arr = np.load(path)
        if arr.ndim == 3 and arr.shape[1] == 17 and arr.shape[2] == 3:
            frames.append(arr.astype(np.float32))
    assert frames, f"no usable (T,17,3) fixtures in {kp_dir}"
    return np.concatenate(frames, axis=0)


def _punch_cycle(
    pool: np.ndarray,
    n_frames: int,
    seed: int,
    amplitude: float = 600.0,
    period: float = 30.0,
) -> np.ndarray:
    """Repeated wrist strikes over a steady stance, as a (T,17,3) sequence.

    The stance has to be *coherent* frame to frame. An earlier version of this
    fixture drew a different real frame each time, which left the pose distance
    from the buffer's anchor permanently high — so ``_stance_cycle_span`` never
    found a return, every profile silently fell through to the trailing fixed
    window, and the stance_cycle path this whole module exists to check was
    never executed. One held base pose with a slow drift, plus a wrist arc whose
    period fits inside the 60-frame rolling buffer, makes the departure and
    return real.
    """
    base_pose = pool[seed % pool.shape[0]].copy()
    seq = np.repeat(base_pose[None, :, :], n_frames, axis=0).copy()

    # Slow whole-body drift, so the subject is not perfectly static.
    drift = np.sin(np.linspace(0.0, 2.0 * np.pi, n_frames, dtype=np.float32))[:, None] * 1.5
    seq[:, :, 0] += drift

    phase = np.linspace(0.0, 2.0 * np.pi * n_frames / period, n_frames, dtype=np.float32)
    arc = (1.0 - np.cos(phase)) * 0.5 * amplitude
    for wrist in (9, 10):
        seq[:, wrist, 0] += arc
        seq[:, wrist, 1] -= arc * 0.4
    seq[:, :, 2] = 0.95
    return seq.astype(np.float32)


def build_cache(
    n_frames: int = 260,
    people: int = 1,
    seed: int = 7,
    drop_pose_frames: set[int] | None = None,
    absent_track_frames: set[int] | None = None,
) -> rc.TrackCache:
    """Assemble a detection table with the awkward cases the replay must handle.

    ``drop_pose_frames`` omits the first pose instance while keeping its box
    (what ``extract_pose_instances`` does to a low-confidence detection).
    ``absent_track_frames`` removes the second person entirely, leaving a stale
    buffer behind.
    """
    pool = _committed_pose_pool()
    drop_pose_frames = drop_pose_frames or set()
    absent_track_frames = absent_track_frames or set()

    per_person = [
        _punch_cycle(pool, n_frames, seed=seed + p, amplitude=600.0 - 180.0 * p, period=30.0 + 7.0 * p)
        for p in range(people)
    ]

    frame_idx, det_order, track_id, box = [], [], [], []
    pose_frame_idx, pose_order, pose_kpts = [], [], []

    for f in range(1, n_frames + 1):
        present = [p for p in range(people) if not (p == 1 and f in absent_track_frames)]
        for slot, p in enumerate(present):
            kp = per_person[p][f - 1]
            xy = kp[:, :2]
            frame_idx.append(f)
            det_order.append(slot)
            track_id.append(p + 1)
            box.append(
                [float(xy[:, 0].min()), float(xy[:, 1].min()),
                 float(xy[:, 0].max()), float(xy[:, 1].max())]
            )
        kept = 0
        for slot, p in enumerate(present):
            if f in drop_pose_frames and slot == 0:
                continue
            pose_frame_idx.append(f)
            pose_order.append(kept)
            pose_kpts.append(per_person[p][f - 1])
            kept += 1

    return rc.TrackCache(
        frame_idx=np.array(frame_idx, dtype=np.int32),
        det_order=np.array(det_order, dtype=np.int16),
        track_id=np.array(track_id, dtype=np.int32),
        box=np.array(box, dtype=np.float32),
        pose_frame_idx=np.array(pose_frame_idx, dtype=np.int32),
        pose_order=np.array(pose_order, dtype=np.int16),
        pose_kpts=np.stack(pose_kpts).astype(np.float32),
        meta={
            "frame_width": 1280,
            "frame_height": 720,
            "fps": 30.0,
            "frames_read": n_frames,
            "reached_eof": True,
            "source": "synthetic",
        },
    )


PROFILES = {
    "punch": dict(
        num_video_sequence_samples=20, reference_sequence_mode="stance_cycle",
        ref_min_motion_energy=0.03, ref_min_return_closure=0.22,
        ref_stance_start_threshold=0.17, ref_stance_end_threshold=0.12,
        ref_stance_peak_threshold=0.30, ref_stance_min_frames=20, ref_stance_hold_frames=4,
    ),
    "kick": dict(
        num_video_sequence_samples=24, reference_sequence_mode="stance_cycle",
        ref_min_motion_energy=0.05, ref_min_return_closure=0.28,
        ref_stance_start_threshold=0.20, ref_stance_end_threshold=0.14,
        ref_stance_peak_threshold=0.38, ref_stance_min_frames=24, ref_stance_hold_frames=4,
    ),
    "fixed": dict(
        num_video_sequence_samples=16, reference_sequence_mode="fixed",
        ref_min_motion_energy=0.0, ref_min_return_closure=0.0,
    ),
    "event": dict(
        num_video_sequence_samples=18, reference_sequence_mode="event_centered",
        ref_min_motion_energy=0.0, ref_min_return_closure=0.0,
    ),
}


def _compare(legacy: dict | None, replayed: list[rs.Candidate], label: str) -> None:
    if legacy is None:
        assert not replayed, f"{label}: replay found {len(replayed)} window(s) where the live path found none"
        print(f"  {label}: both found no candidate")
        return

    assert replayed, f"{label}: live path found a window, replay found none"
    best = replayed[0]

    assert best.end_frame == legacy["frame"], (
        f"{label}: chose a different frame — replay {best.end_frame}, live {legacy['frame']}"
    )
    assert np.array_equal(best.pose_seq, legacy["pose_seq"]), (
        f"{label}: window contents differ (shapes {best.pose_seq.shape} vs {legacy['pose_seq'].shape})"
    )
    for field in ("energy", "closure", "gate_score", "seed_score", "selection_score"):
        mine = float(getattr(best, field))
        theirs = float(legacy[field])
        assert abs(mine - theirs) <= TOL, f"{label}: {field} {mine} != {theirs}"
    print(
        f"  {label}: frame {best.end_frame}, {best.pose_seq.shape[0]} frames, "
        f"selection {best.selection_score:.4f}"
    )


def _top1(cache, config, references, seeds, technique) -> list[rs.Candidate]:
    candidates = rs.iter_candidates(cache, config)
    return rs.select_top_k(
        candidates, 1,
        references=references, seed_references=seeds,
        technique_key=technique, config=config,
        max_self_similarity=0.0,
    )


def test_stance_cycle_path_is_exercised() -> None:
    """Guard against the fixture quietly degenerating to the fallback window.

    Every real capture profile uses stance_cycle, and it is the mode whose
    result depends on buffer alignment. If the synthetic data stops producing a
    genuine departure-and-return, the equivalence checks below still pass — but
    they would be comparing the *fixed* window path, and the mode that matters
    would go untested.
    """
    config = rs.SelectionConfig(**PROFILES["punch"])
    cache = build_cache()
    candidates = rs.iter_candidates(cache, config)
    assert candidates, "fixture produced no candidates at all"
    variable = [c for c in candidates if c.pose_seq.shape[0] != config.num_video_sequence_samples]
    assert variable, (
        "no candidate has a stance-cycle length — every window fell back to the "
        f"trailing {config.num_video_sequence_samples} frames, so the stance_cycle "
        "path is not being tested"
    )
    lengths = sorted({int(c.pose_seq.shape[0]) for c in variable})
    print(f"OK: stance_cycle produced {len(variable)}/{len(candidates)} variable-length window(s), lengths {lengths}")


def test_matches_live_path_bootstrap() -> None:
    """No existing references: ranking falls back to motion energy."""
    print("bootstrap (empty reference bank):")
    for name, profile in PROFILES.items():
        config = rs.SelectionConfig(**profile)
        cache = build_cache()
        legacy = legacy_best_window(cache, config, {}, {}, "jab")
        _compare(legacy, _top1(cache, config, {}, {}, "jab"), name)
    print("OK: replay matches the live path with an empty bank")


def test_matches_live_path_with_reference_bank() -> None:
    """With references present, ranking switches to the DTW gate score.

    That is a different code path — ``base_score`` becomes ``gate_score`` — and
    it is the one the batch actually runs, since the bank is rarely empty.
    """
    print("with a populated reference bank:")
    references = ar.load_reference_pose_library(str(SCRIPT_DIR / "reference_poses"))
    assert references, "expected committed references under reference_poses/"
    technique = "jab" if "jab" in references else sorted(references)[0]

    for name, profile in PROFILES.items():
        config = rs.SelectionConfig(**profile, ref_min_score_gate=0.0)
        cache = build_cache()
        legacy = legacy_best_window(cache, config, references, {}, technique)
        _compare(legacy, _top1(cache, config, references, {}, technique), f"{name} vs {technique}")
    print(f"OK: replay matches the live path against the real {technique} bank")


def test_matches_live_path_two_people() -> None:
    """Two fighters: exercises primary-track selection and its hysteresis."""
    print("two tracked people:")
    config = rs.SelectionConfig(**PROFILES["punch"])
    cache = build_cache(people=2)
    legacy = legacy_best_window(cache, config, {}, {}, "jab")
    _compare(legacy, _top1(cache, config, {}, {}, "jab"), "two people")

    # One fighter leaves for a stretch, so their buffer goes stale while the
    # other keeps accumulating.
    cache = build_cache(people=2, absent_track_frames=set(range(80, 140)))
    legacy = legacy_best_window(cache, config, {}, {}, "jab")
    _compare(legacy, _top1(cache, config, {}, {}, "jab"), "two people, one leaves")
    print("OK: replay matches with multiple tracks and a disappearing one")


def test_matches_live_path_dropped_poses() -> None:
    """Poses dropped by extract_pose_instances must stay dropped, and stay misaligned."""
    print("dropped pose instances:")
    config = rs.SelectionConfig(**PROFILES["punch"])
    cache = build_cache(people=2, drop_pose_frames=set(range(30, 200, 3)))
    legacy = legacy_best_window(cache, config, {}, {}, "jab")
    _compare(legacy, _top1(cache, config, {}, {}, "jab"), "dropped poses")
    print("OK: replay reproduces positional pose/box pairing")


def test_matches_live_path_skip_frame() -> None:
    print("skip_frame > 1:")
    for skip in (2, 3):
        config = rs.SelectionConfig(**PROFILES["punch"], skip_frame=skip)
        cache = build_cache()
        legacy = legacy_best_window(cache, config, {}, {}, "jab")
        _compare(legacy, _top1(cache, config, {}, {}, "jab"), f"skip_frame={skip}")
    print("OK: replay matches with frame skipping")


def test_search_horizon_is_respected() -> None:
    config_full = rs.SelectionConfig(**PROFILES["punch"], search_max_frames=0)
    config_short = rs.SelectionConfig(**PROFILES["punch"], search_max_frames=60)
    cache = build_cache()
    full = rs.iter_candidates(cache, config_full)
    short = rs.iter_candidates(cache, config_short)
    assert len(short) < len(full), "a shorter horizon must consider fewer windows"
    assert all(c.end_frame <= 60 for c in short), "candidates must stay inside the horizon"
    print(f"OK: search horizon respected ({len(full)} candidates full, {len(short)} at 60 frames)")


def test_top_k_windows_are_distinct() -> None:
    """K>1 must return separate moments, not the same action K times."""
    config = rs.SelectionConfig(**PROFILES["punch"])
    cache = build_cache(n_frames=600)
    candidates = rs.iter_candidates(cache, config)
    assert len(candidates) > 4, f"expected several candidates, got {len(candidates)}"

    picked = rs.select_top_k(
        candidates, 4,
        references={}, seed_references={}, technique_key="jab", config=config,
        min_gap_frames=24, max_self_similarity=0.0,
    )
    assert len(picked) >= 2, f"expected multiple windows from one pass, got {len(picked)}"

    for i, a in enumerate(picked):
        for b in picked[i + 1:]:
            a_lo, a_hi = a.span
            b_lo, b_hi = b.span
            assert a_hi < b_lo or b_hi < a_lo, f"windows overlap: {a.span} and {b.span}"
            assert abs(a.center() - b.center()) >= 24, f"windows too close: {a.span} and {b.span}"

    scores = [c.selection_score for c in picked]
    assert scores == sorted(scores, reverse=True), "accepted windows should be in ranked order"
    print(f"OK: top-K returned {len(picked)} disjoint windows from one pass")


def test_top_k_is_deterministic() -> None:
    config = rs.SelectionConfig(**PROFILES["punch"])
    cache = build_cache(n_frames=400)
    runs = []
    for _ in range(3):
        candidates = rs.iter_candidates(cache, config)
        picked = rs.select_top_k(
            candidates, 3,
            references={}, seed_references={}, technique_key="jab", config=config,
        )
        runs.append([(c.start_frame, c.last_frame, round(c.selection_score, 6)) for c in picked])
    assert runs[0] == runs[1] == runs[2], f"selection is not deterministic: {runs}"
    print(f"OK: repeated selection is deterministic ({len(runs[0])} window(s))")


def test_near_duplicate_guard() -> None:
    """The similarity ceiling should reject a second near-identical window."""
    config = rs.SelectionConfig(**PROFILES["punch"])
    cache = build_cache(n_frames=600)
    candidates = rs.iter_candidates(cache, config)

    permissive = rs.select_top_k(
        candidates, 4, references={}, seed_references={}, technique_key="jab",
        config=config, max_self_similarity=0.0,
    )
    strict = rs.select_top_k(
        candidates, 4, references={}, seed_references={}, technique_key="jab",
        config=config, max_self_similarity=1.0,  # reject anything even slightly alike
    )
    assert len(strict) <= len(permissive), "a stricter ceiling cannot accept more windows"
    assert len(strict) == 1, f"an almost-zero ceiling should admit only the first window, got {len(strict)}"
    print(f"OK: near-duplicate guard works ({len(permissive)} permissive, {len(strict)} strict)")


def test_selection_bank_is_not_polluted() -> None:
    """Provisional windows must not leak into the caller's reference library."""
    config = rs.SelectionConfig(**PROFILES["punch"])
    cache = build_cache(n_frames=600)
    references = ar.load_reference_pose_library(str(SCRIPT_DIR / "reference_poses"))
    technique = "jab" if "jab" in references else sorted(references)[0]
    before = {t: set(angles) for t, angles in references.items()}

    rs.select_top_k(
        rs.iter_candidates(cache, config), 4,
        references=references, seed_references={}, technique_key=technique,
        config=config, rescore_after_each=True,
    )
    after = {t: set(angles) for t, angles in references.items()}
    assert before == after, "select_top_k modified the caller's reference bank"
    print("OK: reference bank is left untouched by selection")


def main() -> None:
    test_stance_cycle_path_is_exercised()
    test_matches_live_path_bootstrap()
    test_matches_live_path_with_reference_bank()
    test_matches_live_path_two_people()
    test_matches_live_path_dropped_poses()
    test_matches_live_path_skip_frame()
    test_search_horizon_is_respected()
    test_top_k_windows_are_distinct()
    test_top_k_is_deterministic()
    test_near_duplicate_guard()
    test_selection_bank_is_not_polluted()
    print("\nall capture equivalence checks passed")


if __name__ == "__main__":
    main()
