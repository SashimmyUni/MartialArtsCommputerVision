"""Technique catalogue lookup for the karate reference data.

``reference_poses/karate_techniques.csv`` is the hand-authored catalogue of
karate techniques: romaji, Japanese, English, and — the part the code cares
about — which *family* each technique belongs to (``stance``, ``punch``,
``strike``, ``block``, ``kick``).

The runtime used to infer that family from the technique's English name
("kick" in the name means a kick, and so on). Japanese technique keys such as
``mae_geri`` or ``gyaku_zuki`` carry no such English keyword, so the catalogue
supplies the classification instead and the name heuristics stay as the
fallback for anything not listed.

Deliberately stdlib-only: ``scripts/scout_utils.py`` and
``scripts/run_reference_collection_batch.py`` import this too, and neither
should have to pull in torch to ask what family a technique is.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CATALOG_PATH = PROJECT_ROOT / "reference_poses" / "karate_techniques.csv"

#: Families that group the catalogue's techniques by what the body does.
FAMILIES = ("stance", "punch", "strike", "block", "kick")

_CACHE: dict[str, dict[str, dict[str, str]]] = {}


def normalize_key(text: str) -> str:
    """Canonical snake_case key. Mirrors ``action_recognition._normalize_key``."""
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", (text or "").strip())
    normalized = normalized.replace("-", " ").replace("_", " ")
    return "_".join(normalized.lower().split())


def load_catalog(path: str | Path | None = None) -> dict[str, dict[str, str]]:
    """Load the catalogue keyed by ``technique_key``, including alias keys.

    Aliases (the ``aliases`` column, ``;``-separated) map onto the same row as
    the canonical key, so ``sochin_dachi`` resolves to ``fudo_dachi``. A missing
    catalogue file yields an empty mapping rather than raising — callers fall
    back to their own name heuristics.
    """
    catalog_path = Path(path) if path is not None else CATALOG_PATH
    cache_key = str(catalog_path)
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    entries: dict[str, dict[str, str]] = {}
    if catalog_path.exists():
        with catalog_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                key = normalize_key(row.get("technique_key", ""))
                if not key:
                    continue
                entry = {k: (v or "").strip() for k, v in row.items() if k}
                entry["technique_key"] = key
                entries[key] = entry
                for alias in (row.get("aliases") or "").split(";"):
                    alias_key = normalize_key(alias)
                    if alias_key and alias_key not in entries:
                        entries[alias_key] = entry

    _CACHE[cache_key] = entries
    return entries


@lru_cache(maxsize=512)
def lookup(technique: str) -> dict[str, str] | None:
    """Return the catalogue row for a technique name, or None if unlisted.

    Memoized because ``_technique_angle_category`` asks once per reference per
    scored frame, and the answer never changes within a run.
    """
    return load_catalog().get(normalize_key(technique))


def technique_family(technique: str) -> str | None:
    """Family of a technique (``stance``/``punch``/``strike``/``block``/``kick``)."""
    entry = lookup(technique)
    family = (entry or {}).get("family", "")
    return family if family in FAMILIES else None


def capture_profile_name(technique: str) -> str | None:
    """Capture profile (``stance``/``punch``/``kick``) the batch runner should use."""
    entry = lookup(technique)
    profile = (entry or {}).get("capture_profile", "")
    return profile or None


def focus_joints_group(technique: str) -> str | None:
    """Which body half drives the technique: ``upper``, ``lower`` or ``full``."""
    entry = lookup(technique)
    group = (entry or {}).get("focus_joints", "")
    return group if group in {"upper", "lower", "full"} else None


def search_term(technique: str) -> str | None:
    """Base YouTube search phrase for a technique, without the camera angle."""
    entry = lookup(technique)
    return (entry or {}).get("search_term") or None


def techniques(tier: str | None = None, family: str | None = None) -> list[dict[str, str]]:
    """Catalogue rows in file order, optionally filtered by tier and/or family.

    Alias keys are skipped, so each technique appears exactly once.
    """
    seen: set[int] = set()
    rows: list[dict[str, str]] = []
    for key, entry in load_catalog().items():
        if key != entry.get("technique_key") or id(entry) in seen:
            continue
        seen.add(id(entry))
        if tier and entry.get("tier") != tier:
            continue
        if family and entry.get("family") != family:
            continue
        rows.append(entry)
    return rows


def technique_keys(tier: str | None = None, family: str | None = None) -> list[str]:
    """Canonical technique keys, optionally filtered by tier and/or family."""
    return [entry["technique_key"] for entry in techniques(tier=tier, family=family)]


def classifier_labels(tier: str | None = None) -> list[str]:
    """Zero-shot video-classifier labels for the catalogue, de-duplicated."""
    labels: list[str] = []
    for entry in techniques(tier=tier):
        label = entry.get("classifier_label", "")
        if label and label not in labels:
            labels.append(label)
    return labels


if __name__ == "__main__":  # quick inventory
    catalog = techniques()
    print(f"{len(catalog)} techniques in {CATALOG_PATH}")
    for family in FAMILIES:
        keys = technique_keys(family=family)
        core = technique_keys(tier="core", family=family)
        print(f"  {family:7} {len(keys):3} ({len(core)} core): {', '.join(core)}")
