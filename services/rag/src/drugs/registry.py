"""
registry.py — Curated drug list for MediQuery interaction screening.

Initial scope: Metformin + Lisinopril (FDA labels in knowledge base).
Add entries to registry.json when new drug PDFs are indexed.
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


def all_drug_ids() -> list[str]:
    return list(_load_raw().get("drugs", {}).keys())


def get_drug(drug_id: str) -> dict | None:
    return _load_raw().get("drugs", {}).get(drug_id)


def get_profile(drug_id: str) -> dict | None:
    """Public profile for DrugCard / DrugProfile UI."""
    entry = get_drug(drug_id)
    if not entry:
        return None
    return {
        "id": drug_id,
        "canonical_name": entry.get("canonical_name", drug_id.title()),
        "display_name": entry.get("display_name", entry.get("canonical_name", drug_id.title())),
        "source_pdf": entry.get("source_pdf", ""),
        "indication": entry.get("indication", ""),
        "drug_class": entry.get("drug_class", ""),
        "role": entry.get("role", ""),
        "max_dose": entry.get("max_dose", ""),
        "contraindications": entry.get("contraindications", []),
        "common_adverse_effects": entry.get("common_adverse_effects", []),
    }


def list_profiles(drug_ids: list[str] | None = None) -> list[dict]:
    ids = drug_ids if drug_ids is not None else all_drug_ids()
    profiles = []
    for drug_id in ids:
        profile = get_profile(drug_id)
        if profile:
            profiles.append(profile)
    return profiles


def drugs_in_knowledge_base(docs_dir: Path | None = None) -> list[str]:
    """
    Return drug IDs whose source_pdf exists in documents/ (or all registered
    drugs when the folder is unavailable — e.g. empty dev checkout).
    """
    registered = all_drug_ids()
    if docs_dir is None:
        return registered

    if not docs_dir.is_dir():
        return registered

    pdf_names = {p.name.lower() for p in docs_dir.glob("**/*.pdf")}
    in_kb = []
    for drug_id in registered:
        entry = get_drug(drug_id)
        if not entry:
            continue
        source = entry.get("source_pdf", "")
        if source.lower() in pdf_names:
            in_kb.append(drug_id)
    return in_kb if in_kb else registered


def alias_to_drug_id() -> dict[str, str]:
    """Map lowercase alias → canonical drug id."""
    mapping: dict[str, str] = {}
    for drug_id, entry in _load_raw().get("drugs", {}).items():
        mapping[drug_id.lower()] = drug_id
        for alias in entry.get("aliases", []):
            mapping[alias.lower().strip()] = drug_id
        canonical = entry.get("canonical_name", "")
        if canonical:
            mapping[canonical.lower().strip()] = drug_id
    return mapping


def source_pdf_to_drug_id() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for drug_id, entry in _load_raw().get("drugs", {}).items():
        source = entry.get("source_pdf", "")
        if source:
            mapping[source.lower()] = drug_id
    return mapping


def normalize_drug_id(name: str) -> str | None:
    """Resolve user input or alias to a registry drug id."""
    if not name or not name.strip():
        return None
    key = name.strip().lower()
    return alias_to_drug_id().get(key)


_WORD_BOUNDARY = r"(?<![a-z0-9])"


def _alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias.lower())
    return re.compile(rf"{_WORD_BOUNDARY}{escaped}(?![a-z0-9])", re.IGNORECASE)


@lru_cache(maxsize=1)
def _compiled_alias_patterns() -> list[tuple[str, re.Pattern[str]]]:
    patterns: list[tuple[str, re.Pattern[str]]] = []
    seen: set[str] = set()
    for alias, drug_id in sorted(
        alias_to_drug_id().items(), key=lambda x: (-len(x[0]), x[0])
    ):
        if alias in seen:
            continue
        seen.add(alias)
        if len(alias) < 3 and alias not in ("ace",):
            continue
        patterns.append((drug_id, _alias_pattern(alias)))
    return patterns


def find_drugs_in_text(text: str, *, allowed_ids: list[str] | None = None) -> list[str]:
    """Return unique drug ids mentioned in text (longest-alias match first)."""
    if not text or not text.strip():
        return []

    allowed = set(allowed_ids) if allowed_ids else None
    found: list[str] = []
    seen: set[str] = set()

    for drug_id, pattern in _compiled_alias_patterns():
        if allowed is not None and drug_id not in allowed:
            continue
        if drug_id in seen:
            continue
        if pattern.search(text):
            found.append(drug_id)
            seen.add(drug_id)

    return found
