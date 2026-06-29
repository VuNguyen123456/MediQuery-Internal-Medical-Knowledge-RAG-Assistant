"""
precautions.py — Document-grounded vaccine precaution screening.

Hybrid retrieval per vaccine × condition:
  1. Metadata-filtered chunks tagged with vaccine id
  2. Semantic search for precaution/contraindication language
  3. Per-schedule-PDF retrieval from registry

Tiers (document-grounded only):
  red    — contraindication / defer language in retrieved excerpts
  yellow — precaution / consult language without clear contraindication
  green  — relevant chunks found, no precaution language
  grey   — insufficient relevant excerpts in indexed sources
"""

from __future__ import annotations

import re
from typing import Literal

from generation.prompt import _format_excerpt
from retrieval.search import retrieve, retrieve_filtered, retrieve_for_source
from screening.common import chunks_to_citations, merge_chunks
from vaccines.registry import find_vaccines_in_text, get_vaccine

PrecautionTier = Literal["red", "yellow", "green", "grey"]

DISCLAIMER = (
    "Vaccine precaution screening is based on indexed schedule documents only — "
    "not a substitute for clinical judgment, immunization information systems, "
    "or up-to-date CDC/ACIP guidance."
)

RED_PATTERNS = [
    r"\bcontraindicat",
    r"\bdo not (give|administer|vaccinate)\b",
    r"\bshould not (receive|be vaccinated|get)\b",
    r"\bdefer vaccination\b",
    r"\bsevere allergic",
    r"\banaphylaxis\b",
    r"\bnot recommended\b",
]

YELLOW_PATTERNS = [
    r"\bprecaution[s]?\b",
    r"\bconsult\b",
    r"\bimmunocompromis",
    r"\begg allergy\b",
    r"\bpregnan",
    r"\bcaution\b",
    r"\bdefer\b",
    r"\bwait\b",
    r"\bmedical consultation\b",
]

_red_re = re.compile("|".join(RED_PATTERNS), re.IGNORECASE)
_yellow_re = re.compile("|".join(YELLOW_PATTERNS), re.IGNORECASE)


def _display_name(vaccine_id: str) -> str:
    entry = get_vaccine(vaccine_id)
    if entry:
        return entry.get("display_name") or entry.get("canonical_name") or vaccine_id.title()
    return vaccine_id.title()


def _check_label(vaccine_id: str, condition: str) -> str:
    return f"{_display_name(vaccine_id)} × {condition.strip()}"


def _classify_tier(chunks: list[dict]) -> PrecautionTier:
    if not chunks:
        return "grey"

    combined_text = " ".join(c.get("text", "") for c in chunks)
    if _red_re.search(combined_text):
        return "red"
    if _yellow_re.search(combined_text):
        return "yellow"
    return "green"


def _tier_message(tier: PrecautionTier, vaccine_id: str, condition: str) -> str:
    label = _check_label(vaccine_id, condition)
    if tier == "red":
        return f"Contraindication or deferral language found in indexed documents for {label}."
    if tier == "yellow":
        return f"Precaution language found in indexed documents for {label} — review citations."
    if tier == "green":
        return f"No precaution language found in indexed documents for {label}."
    return f"Insufficient data in indexed documents to assess {label}."


def _build_summary(tier: PrecautionTier, chunks: list[dict], vaccine_id: str, condition: str) -> str:
    if tier == "grey":
        return _tier_message(tier, vaccine_id, condition)

    if tier == "green":
        return (
            f"No significant precaution language found in indexed documents for "
            f"{_check_label(vaccine_id, condition)}. This does not confirm eligibility — "
            "consult source schedules."
        )

    best = chunks[0]
    excerpt = _format_excerpt(best.get("text", ""), max_len=280)
    if excerpt:
        return excerpt
    return _tier_message(tier, vaccine_id, condition)


def _chunk_mentions_vaccine(chunk: dict, vaccine_id: str) -> bool:
    vaccines = chunk.get("vaccines")
    if isinstance(vaccines, list) and vaccine_id in vaccines:
        return True
    return vaccine_id in find_vaccines_in_text(chunk.get("text", ""))


def _retrieve_vaccine_chunks(vaccine_id: str, condition: str) -> list[dict]:
    """Hybrid retrieval for vaccine × condition."""
    name = _display_name(vaccine_id)
    condition = condition.strip()

    metadata_filtered = retrieve_filtered(
        f"{name} vaccine {condition} contraindication precaution",
        metadata_filter={"vaccines": {"$in": [vaccine_id]}},
        top_k=4,
    )

    semantic = retrieve(
        f"{name} vaccine {condition} contraindication precaution defer eligibility",
        top_k=6,
    )
    semantic = [c for c in semantic if _chunk_mentions_vaccine(c, vaccine_id)]

    entry = get_vaccine(vaccine_id) or {}
    schedule_pdfs = entry.get("schedule_pdfs", [])
    per_pdf: list[dict] = []
    for pdf in schedule_pdfs:
        per_pdf.extend(
            retrieve_for_source(
                f"{name} {condition} contraindication precaution vaccine schedule",
                source_pdf=pdf,
                top_k=3,
            )
        )

    return merge_chunks(metadata_filtered, semantic, per_pdf)


def screen_vaccine(vaccine_id: str, condition: str) -> dict:
    if not condition or not condition.strip():
        raise ValueError("Condition is required for vaccine precaution screening")

    chunks = _retrieve_vaccine_chunks(vaccine_id, condition)
    tier = _classify_tier(chunks)

    return {
        "vaccine_id": vaccine_id,
        "condition": condition.strip(),
        "label": _check_label(vaccine_id, condition),
        "tier": tier,
        "summary": _build_summary(tier, chunks, vaccine_id, condition),
        "citations": chunks_to_citations(chunks),
    }


def screen_vaccines(vaccine_ids: list[str], condition: str) -> dict:
    """
    Screen each vaccine against the given patient condition/context.

    condition is required (session context from UI).
    """
    if not condition or not condition.strip():
        raise ValueError("Condition is required for vaccine precaution screening")

    unique: list[str] = []
    seen: set[str] = set()
    for vaccine_id in vaccine_ids:
        normalized = vaccine_id.lower().strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)

    checks = [screen_vaccine(v, condition) for v in unique]

    return {
        "checks": checks,
        "disclaimer": DISCLAIMER,
        "vaccines_screened": unique,
        "condition": condition.strip(),
    }
