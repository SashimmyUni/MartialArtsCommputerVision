"""Check the cross-cut fighter identity registry in fighter_identity.py.

This is the correctness gate for "who threw this technique, fighter 1 or
fighter 2" holding up across a whole fight: YOLO/ByteTrack's track_id resets
on every broadcast cut, so identity must be re-derived from appearance
instead. Tests operate on synthetic embedding vectors directly (not real
crops/cv2 histograms) so they need only numpy and stay fast.

Run from anywhere:

    python test_fighter_identity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from fighter_identity import FighterIdentityRegistry, histogram_similarity, torso_histogram  # noqa: E402

RED_CORNER = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
BLUE_CORNER = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def test_first_two_tracks_become_distinct_slots() -> None:
    reg = FighterIdentityRegistry(max_fighters=2)
    slot_a = reg.resolve(track_id=1, embedding=RED_CORNER)
    slot_b = reg.resolve(track_id=2, embedding=BLUE_CORNER)
    assert slot_a != slot_b, "two dissimilar appearances must get distinct slots"
    assert set(reg.slots) == {slot_a, slot_b}
    print("OK: first two dissimilar tracks become distinct slots")


def test_same_track_id_keeps_its_slot() -> None:
    reg = FighterIdentityRegistry(max_fighters=2)
    slot_a1 = reg.resolve(track_id=1, embedding=RED_CORNER)
    slot_a2 = reg.resolve(track_id=1, embedding=RED_CORNER * 0.9)
    assert slot_a1 == slot_a2, "the same track_id must never change slot"
    print("OK: repeated resolve() on one track_id keeps its slot")


def test_identity_survives_track_id_reset_across_a_cut() -> None:
    """The scenario this module exists for: a cut drops track_id=1 and the
    fighter reappears as track_id=7, but still looks the same.
    """
    reg = FighterIdentityRegistry(max_fighters=2, match_threshold=0.55)
    red_slot = reg.resolve(track_id=1, embedding=RED_CORNER)
    reg.resolve(track_id=2, embedding=BLUE_CORNER)
    reg.forget(track_id=1)  # the track is gone after the cut

    same_fighter_new_id = reg.resolve(track_id=7, embedding=RED_CORNER * 0.95)
    assert same_fighter_new_id == red_slot, "a familiar appearance under a new track_id must rejoin its slot"
    print("OK: fighter identity survives a track_id reset across a cut")


def test_gallery_capped_at_max_fighters() -> None:
    """A referee/cornerman wandering into frame must not create a 3rd slot."""
    reg = FighterIdentityRegistry(max_fighters=2, match_threshold=0.9)
    reg.resolve(track_id=1, embedding=RED_CORNER)
    reg.resolve(track_id=2, embedding=BLUE_CORNER)
    referee = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # dissimilar to both corners
    third_slot = reg.resolve(track_id=3, embedding=referee)
    assert len(reg.slots) == 2, f"gallery must stay capped at 2, got {reg.slots}"
    assert third_slot in reg.slots
    print("OK: gallery stays capped at max_fighters, best-effort-assigning overflow")


def test_dissimilar_appearance_creates_new_slot_when_room_remains() -> None:
    reg = FighterIdentityRegistry(max_fighters=2, match_threshold=0.9)
    reg.resolve(track_id=1, embedding=RED_CORNER)
    slot_b = reg.resolve(track_id=2, embedding=BLUE_CORNER)
    assert len(reg.slots) == 2
    assert reg.last_match_score[2] == 1.0
    assert slot_b in reg.slots
    print("OK: a dissimilar appearance takes the remaining free slot")


def test_histogram_similarity_ranks_identical_over_different() -> None:
    identical = histogram_similarity(RED_CORNER, RED_CORNER)
    different = histogram_similarity(RED_CORNER, BLUE_CORNER)
    assert identical > different, (identical, different)
    assert histogram_similarity(np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)) == -1.0
    print("OK: histogram_similarity ranks identical appearance above different")


def test_torso_histogram_handles_degenerate_crops() -> None:
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    hist = torso_histogram(empty)
    assert hist.shape == (16 * 16,)
    assert float(hist.sum()) == 0.0

    solid_red = np.zeros((100, 60, 3), dtype=np.uint8)
    solid_red[:, :, 2] = 255  # BGR -> pure red
    hist2 = torso_histogram(solid_red)
    assert hist2.shape == (16 * 16,)
    assert abs(float(hist2.sum()) - 1.0) < 1e-4, "a non-degenerate crop's histogram should be L1-normalized"
    print("OK: torso_histogram handles empty crops and normalizes real ones")


if __name__ == "__main__":
    test_first_two_tracks_become_distinct_slots()
    test_same_track_id_keeps_its_slot()
    test_identity_survives_track_id_reset_across_a_cut()
    test_gallery_capped_at_max_fighters()
    test_dissimilar_appearance_creates_new_slot_when_room_remains()
    test_histogram_similarity_ranks_identical_over_different()
    test_torso_histogram_handles_degenerate_crops()
    print("\nOK: all fighter_identity tests passed")
