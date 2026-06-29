"""
registry.py — Curated vaccine list for MediQuery precaution screening.

Initial scope: common vaccines referenced in CDC schedule PDFs under documents/vaccines/.
Add entries to registry.json when new schedule documents are indexed.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_REGISTRY_PATH = Path(__file__).resolve().parent / "registry.json"


@lru_cache(maxsize=1)
def _load_raw() -> dict:
    with _REGISTRY_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def all_vaccine_ids() -> list[str]:
    return list(_load_raw().get("vaccines", {}).keys())


def get_vaccine(vaccine_id: str) -> dict | None:
    return _load_raw().get("vaccines", {}).get(vaccine_id)


def get_profile(vaccine_id: str) -> dict | None:
    """Public profile for VaccineCard UI."""
    entry = get_vaccine(vaccine_id)
    if not entry:
        return None
    return {
        "id": vaccine_id,
        "canonical_name": entry.get("canonical_name", vaccine_id.title()),
        "display_name": entry.get("display_name", entry.get("canonical_name", vaccine_id.title())),
        "schedule_pdfs": entry.get("schedule_pdfs", []),
        "age_groups": entry.get("age_groups", []),
        "common_precautions": entry.get("common_precautions", []),
    }


def list_profiles(vaccine_ids: list[str] | None = None) -> list[dict]:
    ids = vaccine_ids if vaccine_ids is not None else all_vaccine_ids()
    profiles = []
    for vaccine_id in ids:
        profile = get_profile(vaccine_id)
        if profile:
            profiles.append(profile)
    return profiles


def vaccines_in_knowledge_base(docs_dir: Path | None = None) -> list[str]:
    """
    Return vaccine IDs whose schedule_pdfs exist under documents/vaccines/
    (or all registered vaccines when the folder is unavailable).
    """
    registered = all_vaccine_ids()
    if docs_dir is None:
        return registered

    vaccines_dir = docs_dir / "vaccines"
    if not vaccines_dir.is_dir():
        vaccines_dir = docs_dir
    if not vaccines_dir.is_dir():
        return registered

    pdf_names = {p.name.lower() for p in vaccines_dir.glob("**/*.pdf")}
    in_kb = []
    for vaccine_id in registered:
        entry = get_vaccine(vaccine_id)
        if not entry:
            continue
        schedule_pdfs = entry.get("schedule_pdfs", [])
        if any(pdf.lower() in pdf_names for pdf in schedule_pdfs):
            in_kb.append(vaccine_id)
    return in_kb if in_kb else registered


def alias_to_vaccine_id() -> dict[str, str]:
    """Map lowercase alias → canonical vaccine id."""
    mapping: dict[str, str] = {}
    for vaccine_id, entry in _load_raw().get("vaccines", {}).items():
        mapping[vaccine_id.lower()] = vaccine_id
        for alias in entry.get("aliases", []):
            mapping[alias.lower().strip()] = vaccine_id
        canonical = entry.get("canonical_name", "")
        if canonical:
            mapping[canonical.lower().strip()] = vaccine_id
    return mapping


def normalize_vaccine_id(name: str) -> str | None:
    """Resolve user input or alias to a registry vaccine id."""
    if not name or not name.strip():
        return None
    key = name.strip().lower()
    return alias_to_vaccine_id().get(key)


_WORD_BOUNDARY = r"(?<![a-z0-9])"


def _alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias.lower())
    return re.compile(rf"{_WORD_BOUNDARY}{escaped}(?![a-z0-9])", re.IGNORECASE)


@lru_cache(maxsize=1)
def _compiled_alias_patterns() -> list[tuple[str, re.Pattern[str]]]:
    patterns: list[tuple[str, re.Pattern[str]]] = []
    seen: set[str] = set()
    for alias, vaccine_id in sorted(
        alias_to_vaccine_id().items(), key=lambda x: (-len(x[0]), x[0])
    ):
        if alias in seen:
            continue
        seen.add(alias)
        if len(alias) < 3 and alias not in ("flu",):
            continue
        patterns.append((vaccine_id, _alias_pattern(alias)))
    return patterns


def find_vaccines_in_text(text: str, *, allowed_ids: list[str] | None = None) -> list[str]:
    """Return unique vaccine ids mentioned in text (longest-alias match first)."""
    if not text or not text.strip():
        return []

    allowed = set(allowed_ids) if allowed_ids else None
    found: list[str] = []
    seen: set[str] = set()

    for vaccine_id, pattern in _compiled_alias_patterns():
        if allowed is not None and vaccine_id not in allowed:
            continue
        if vaccine_id in seen:
            continue
        if pattern.search(text):
            found.append(vaccine_id)
            seen.add(vaccine_id)

    return found
