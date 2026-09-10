"""Technique catalogue lookup, primarily for the karate reference data.

``reference_poses/karate_techniques.csv`` is the hand-authored catalogue of
karate techniques: romaji, Japanese, English, and — the part the code cares
about — which *family* each technique belongs to (``stance``, ``punch``,
``strike``, ``block``, ``kick``).

The runtime used to infer that family from the technique's English name
("kick" in the name means a kick, and so on). Japanese technique keys such as
``mae_geri`` or ``gyaku_zuki`` carry no such English keyword, so the catalogue
supplies the classification instead and the name heuristics stay as the
fallback for anything not listed.

``reference_poses/mma_techniques.csv`` is a second catalogue (standing MMA
striking, for ``fight_analysis.py``) in the same schema. It is deliberately
**not** merged into the unpathed/default lookups (``lookup``,
``technique_family``, ``focus_joints_group``, ...): several of its rows reuse
technique keys the kickboxing trainer already scores against today (``jab``,
``uppercut``, ``front_kick``, ...), which today resolve as "unlisted" and fall
back to this module's name heuristics. Folding the mma catalogue into the
default would make those keys newly resolve through it instead, silently
changing which joints the *existing* trainer compares for them. Callers that
want the mma catalogue must pass ``path=MMA_CATALOG_PATH`` (or
``path=ALL_CATALOG_PATHS`` for both) explicitly — see ``fight_analysis.py``.

Deliberately stdlib-only: ``scripts/scout_utils.py`` and
``scripts/run_reference_collection_batch.py`` import this too, and neither
should have to pull in torch to ask what family a technique is.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
KARATE_CATALOG_PATH = PROJECT_ROOT / "reference_poses" / "karate_techniques.csv"
MMA_CATALOG_PATH = PROJECT_ROOT / "reference_poses" / "mma_techniques.csv"
#: Back-compat alias — this used to be the only catalogue file.
CATALOG_PATH = KARATE_CATALOG_PATH
#: What an unpathed lookup uses — karate only, see the module docstring.
DEFAULT_CATALOG_PATHS: tuple[Path, ...] = (KARATE_CATALOG_PATH,)
#: Both catalogues, for callers that explicitly want the merged view
#: (e.g. this module's own ``__main__`` inventory).
ALL_CATALOG_PATHS: tuple[Path, ...] = (KARATE_CATALOG_PATH, MMA_CATALOG_PATH)

#: Families that group the catalogue's techniques by what the body does.
FAMILIES = ("stance", "punch", "strike", "block", "kick")

_CACHE: dict[str, dict[str, dict[str, str]]] = {}


def normalize_key(text: str) -> str:
    """Canonical snake_case key. Mirrors ``action_recognition._normalize_key``."""
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", (text or "").strip())
    normalized = normalized.replace("-", " ").replace("_", " ")
    return "_".join(normalized.lower().split())


def _load_one_catalog(catalog_path: Path) -> dict[str, dict[str, str]]:
    """Load a single catalogue CSV keyed by ``technique_key``, including alias keys.

    Aliases (the ``aliases`` column, ``;``-separated) map onto the same row as
    the canonical key, so ``sochin_dachi`` resolves to ``fudo_dachi``. A missing
    catalogue file yields an empty mapping rather than raising — callers fall
    back to their own name heuristics.
    """
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
    return entries


def load_catalog(path: str | Path | Sequence[str | Path] | None = None) -> dict[str, dict[str, str]]:
    """Load the catalogue keyed by ``technique_key``, including alias keys.

    ``path`` may be a single file, a sequence of files (merged in order — the
    first file's row wins a ``technique_key`` collision), or omitted to use
    ``DEFAULT_CATALOG_PATHS`` (karate only — see the module docstring for why
    the mma catalogue is not folded in here). A missing catalogue file
    contributes no rows rather than raising — callers fall back to their own
    name heuristics.
    """
    if path is None:
        catalog_paths: tuple[Path, ...] = DEFAULT_CATALOG_PATHS
    elif isinstance(path, (str, Path)):
        catalog_paths = (Path(path),)
    else:
        catalog_paths = tuple(Path(p) for p in path)

    cache_key = "|".join(str(p) for p in catalog_paths)
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    entries: dict[str, dict[str, str]] = {}
    for catalog_path in catalog_paths:
        for key, entry in _load_one_catalog(catalog_path).items():
            entries.setdefault(key, entry)

    _CACHE[cache_key] = entries
    return entries


@lru_cache(maxsize=512)
def lookup(technique: str, path: str | Path | None = None) -> dict[str, str] | None:
    """Return the catalogue row for a technique name, or None if unlisted.

    Memoized because ``_technique_angle_category`` asks once per reference per
    scored frame, and the answer never changes within a run.
    """
    return load_catalog(path).get(normalize_key(technique))


def technique_family(technique: str, path: str | Path | None = None) -> str | None:
    """Family of a technique (``stance``/``punch``/``strike``/``block``/``kick``)."""
    entry = lookup(technique, path=path)
    family = (entry or {}).get("family", "")
    return family if family in FAMILIES else None


def capture_profile_name(technique: str, path: str | Path | None = None) -> str | None:
    """Capture profile (``stance``/``punch``/``kick``) the batch runner should use."""
    entry = lookup(technique, path=path)
    profile = (entry or {}).get("capture_profile", "")
    return profile or None


def focus_joints_group(technique: str, path: str | Path | None = None) -> str | None:
    """Which body half drives the technique: ``upper``, ``lower`` or ``full``."""
    entry = lookup(technique, path=path)
    group = (entry or {}).get("focus_joints", "")
    return group if group in {"upper", "lower", "full"} else None


#: Folder names under reference_poses/ that predate this catalogue and don't
#: match their catalogue technique_key (folder ``knee_kick`` vs. catalogue
#: ``knee_strike``, folder ``elbow`` vs. ``elbow_strike``). Deliberately kept
#: out of the CSV ``aliases`` column: registering them there would make e.g.
#: ``technique_family("elbow")`` resolve through the new catalogue row instead
#: of staying unlisted, changing which angle joints the pre-existing
#: kickboxing trainer compares for that technique. Consulted only by
#: ``resolve_reference_key``.
LEGACY_REFERENCE_FOLDER_NAMES: dict[str, str] = {
    "elbow_strike": "elbow",
    "knee_strike": "knee_kick",
}


def resolve_reference_key(
    references: dict[str, object], technique: str, path: str | Path | None = None
) -> str | None:
    """Map a canonical technique key onto whatever key a reference bank
    dict (e.g. ``action_recognition.load_reference_pose_library``'s return
    value) actually uses.

    Tries, in order: the key itself, ``LEGACY_REFERENCE_FOLDER_NAMES``, then
    any catalogue ``aliases`` for that key (looked up via ``path`` — pass
    ``MMA_CATALOG_PATH`` to resolve mma technique keys). Returns ``None`` if
    none of those has a bank.
    """
    key = normalize_key(technique)
    if key in references:
        return key
    legacy = LEGACY_REFERENCE_FOLDER_NAMES.get(key)
    if legacy and legacy in references:
        return legacy
    entry = lookup(key, path=path)
    for alias in (entry or {}).get("aliases", "").split(";"):
        alias_key = normalize_key(alias)
        if alias_key and alias_key in references:
            return alias_key
    return None


def search_term(technique: str, path: str | Path | None = None) -> str | None:
    """Base YouTube search phrase for a technique, without the camera angle."""
    entry = lookup(technique, path=path)
    return (entry or {}).get("search_term") or None


def techniques(
    tier: str | None = None,
    family: str | None = None,
    path: str | Path | Sequence[str | Path] | None = None,
) -> list[dict[str, str]]:
    """Catalogue rows in file order, optionally filtered by tier and/or family.

    Alias keys are skipped, so each technique appears exactly once. ``path``
    behaves as in ``load_catalog`` — omit it to use the karate-only default.
    """
    seen: set[int] = set()
    rows: list[dict[str, str]] = []
    for key, entry in load_catalog(path).items():
        if key != entry.get("technique_key") or id(entry) in seen:
            continue
        seen.add(id(entry))
        if tier and entry.get("tier") != tier:
            continue
        if family and entry.get("family") != family:
            continue
        rows.append(entry)
    return rows


def technique_keys(
    tier: str | None = None,
    family: str | None = None,
    path: str | Path | Sequence[str | Path] | None = None,
) -> list[str]:
    """Canonical technique keys, optionally filtered by tier and/or family."""
    return [entry["technique_key"] for entry in techniques(tier=tier, family=family, path=path)]


def classifier_labels(tier: str | None = None, path: str | Path | Sequence[str | Path] | None = None) -> list[str]:
    """Zero-shot video-classifier labels for the catalogue, de-duplicated."""
    labels: list[str] = []
    for entry in techniques(tier=tier, path=path):
        label = entry.get("classifier_label", "")
        if label and label not in labels:
            labels.append(label)
    return labels


if __name__ == "__main__":  # quick inventory (both catalogues, shown explicitly-merged)
    catalog = techniques(path=ALL_CATALOG_PATHS)
    paths = ", ".join(str(p) for p in ALL_CATALOG_PATHS)
    print(f"{len(catalog)} techniques merged from: {paths}")
    for family in FAMILIES:
        keys = technique_keys(family=family, path=ALL_CATALOG_PATHS)
        core = technique_keys(tier="core", family=family, path=ALL_CATALOG_PATHS)
        print(f"  {family:7} {len(keys):3} ({len(core)} core): {', '.join(core)}")
