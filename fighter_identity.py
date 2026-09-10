"""Cross-cut fighter identity: map ephemeral YOLO track_ids to persistent
fighter slots so identity survives a broadcast cut.

YOLO/ByteTrack's ``track_id`` resets whenever a track is lost and
re-acquired -- which happens on every camera cut in broadcast footage. For
"who threw this technique, fighter 1 or fighter 2" statistics to hold up
across a whole fight, identity needs to be anchored to something that
survives a cut. This module uses a cheap appearance embedding (an HSV color
histogram of the fighter's torso/shorts region, where corner color is most
consistent) matched against a small capped gallery -- not a trained
person-ReID model. It is explicitly best-effort: similar-colored gear, a
referee, or a cornerman briefly in frame can be misassigned.
"""

from __future__ import annotations

import numpy as np


def torso_histogram(crop: np.ndarray, bins: int = 16, min_saturation: int = 40) -> np.ndarray:
    """HSV hue/saturation histogram of a person crop's torso band, L1-normalized.

    Uses the crop's vertical-middle band (roughly torso/shorts, avoiding the
    head and feet) as a cheap proxy for corner color -- the most consistent
    appearance cue for one fighter across a whole match. Pixels below
    ``min_saturation`` are dropped before histogramming: a bounding-box crop
    always includes some background/mat/canvas, which is usually low-
    saturation gray and would otherwise dominate the histogram and make two
    different fighters on the same background read as similar. Returns a
    flat ``(bins*bins,)`` float32 vector; an empty/degenerate/all-gray crop
    returns zeros.
    """
    import cv2

    if crop is None or crop.size == 0:
        return np.zeros(bins * bins, dtype=np.float32)

    h, w = crop.shape[:2]
    y0, y1 = int(h * 0.35), int(h * 0.85)
    band = crop[max(0, y0) : max(y0 + 1, y1), :]
    if band.size == 0:
        return np.zeros(bins * bins, dtype=np.float32)

    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 1] >= min_saturation).astype(np.uint8) * 255
    hist = cv2.calcHist([hsv], [0, 1], mask, [bins, bins], [0, 180, 0, 256])
    total = float(hist.sum())
    if total > 1e-9:
        hist = hist / total
    return hist.flatten().astype(np.float32)


def histogram_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation-based similarity in ``[-1, 1]``; ``1.0`` means identical
    distributions. Degenerate (empty/mismatched-shape) inputs score ``-1.0``
    so they never win a best-match comparison.
    """
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return -1.0
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = float(np.linalg.norm(a_c) * np.linalg.norm(b_c))
    if denom < 1e-9:
        # Both (near-)constant vectors: call it identical only if their means matched too.
        return 1.0 if abs(float(a.mean()) - float(b.mean())) < 1e-9 else 0.0
    return float(np.dot(a_c, b_c) / denom)


class FighterIdentityRegistry:
    """Assigns persistent fighter slots (``fighter_1``, ``fighter_2``, ...) to
    track_ids by appearance, capped at ``max_fighters`` gallery entries.

    A track_id seen before keeps its slot for free. A brand-new track_id is
    matched against the gallery by histogram similarity: a good match
    (``>= match_threshold``) inherits that slot -- this is what lets identity
    survive a track_id reset across a cut. A poor match becomes a new slot if
    the gallery has room; otherwise (best-effort) it still joins the closest
    slot so every tracked person gets an identity, and the caller can inspect
    ``last_match_score`` to see how confident that assignment was.
    """

    def __init__(
        self,
        max_fighters: int = 2,
        match_threshold: float = 0.55,
        embedding_momentum: float = 0.8,
    ) -> None:
        if max_fighters < 1:
            raise ValueError("max_fighters must be >= 1")
        if not (0.0 <= embedding_momentum < 1.0):
            raise ValueError("embedding_momentum must be in [0.0, 1.0)")
        self.max_fighters = max_fighters
        self.match_threshold = match_threshold
        self.embedding_momentum = embedding_momentum
        self._gallery: dict[str, np.ndarray] = {}
        self._track_to_slot: dict[int, str] = {}
        self.last_match_score: dict[int, float] = {}

    @property
    def slots(self) -> list[str]:
        return list(self._gallery.keys())

    def resolve(self, track_id: int, embedding: np.ndarray) -> str:
        """Return the persistent fighter slot for ``track_id``, updating the
        gallery with ``embedding`` (that track's latest appearance).
        """
        existing_slot = self._track_to_slot.get(track_id)
        if existing_slot is not None:
            self._update_gallery(existing_slot, embedding)
            self.last_match_score[track_id] = 1.0
            return existing_slot

        if not self._gallery:
            return self._new_slot(track_id, embedding)

        best_slot, best_score = self._best_match(embedding)
        if best_score >= self.match_threshold or len(self._gallery) >= self.max_fighters:
            self._track_to_slot[track_id] = best_slot
            self._update_gallery(best_slot, embedding)
            self.last_match_score[track_id] = best_score
            return best_slot

        return self._new_slot(track_id, embedding)

    def forget(self, track_id: int) -> None:
        """Drop a track_id's slot mapping.

        The gallery entry itself is untouched -- a *different* track_id
        reassociating with that slot by appearance after a cut is the entire
        point of this registry.
        """
        self._track_to_slot.pop(track_id, None)
        self.last_match_score.pop(track_id, None)

    def _new_slot(self, track_id: int, embedding: np.ndarray) -> str:
        slot = f"fighter_{len(self._gallery) + 1}"
        self._gallery[slot] = np.asarray(embedding, dtype=np.float32).copy()
        self._track_to_slot[track_id] = slot
        self.last_match_score[track_id] = 1.0
        return slot

    def _update_gallery(self, slot: str, embedding: np.ndarray) -> None:
        m = self.embedding_momentum
        self._gallery[slot] = (m * self._gallery[slot] + (1.0 - m) * np.asarray(embedding, dtype=np.float32)).astype(
            np.float32
        )

    def _best_match(self, embedding: np.ndarray) -> tuple[str, float]:
        scored = [(histogram_similarity(embedding, emb), slot) for slot, emb in self._gallery.items()]
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_slot = scored[0]
        return best_slot, best_score
