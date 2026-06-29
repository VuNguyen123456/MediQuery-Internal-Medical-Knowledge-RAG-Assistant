"""
precautions.py — Document-grounded drug precaution screening (drug × patient context).

Hybrid retrieval per drug × condition:
  1. Metadata-filtered chunks tagged with drug id
  2. Semantic search for contraindication / precaution language
  3. Per-label-PDF retrieval from registry

Tiers match vaccine precaution screening (document-grounded regex).
"""

from __future__ import annotations

import re
from typing import Literal

from drugs.registry import find_drugs_in_text, get_drug
from generation.prompt import _format_excerpt
from retrieval.search import retrieve, retrieve_filtered, retrieve_for_source
from screening.common import chunks_to_citations, merge_chunks

PrecautionTier = Literal["red", "yellow", "green", "grey"]

DISCLAIMER = (
    "Drug precaution screening is based on indexed label documents only — "
    "not a substitute for pharmacist review, prescribing information, or clinical judgment."
)

RED_PATTERNS = [
    r"\bcontraindicat",
    r"\bdo not use\b",
    r"\bavoid\b",
    r"\bshould not\b",
    r"\bnot recommended\b",
    r"\bdiscontinue\b",
    r"\bsevere allergic",
    r"\banaphylaxis\b",
]

YELLOW_PATTERNS = [
    r"\bprecaution[s]?\b",
    r"\bconsult\b",
    r"\bcaution\b",
    r"\bwarning[s]?\b",
    r"\bmonitor\b",
    r"\bpregnan",
    r"\blactation\b",
    r"\brenal impairment\b",
    r"\bhepatic impairment\b",
    r"\bincreased risk\b",
]

_red_re = re.compile("|".join(RED_PATTERNS), re.IGNORECASE)
_yellow_re = re.compile("|".join(YELLOW_PATTERNS), re.IGNORECASE)


def _display_name(drug_id: str) -> str:
    entry = get_drug(drug_id)
    if entry:
        return entry.get("display_name") or entry.get("canonical_name") or drug_id.title()
    return drug_id.title()


def _check_label(drug_id: str, condition: str) -> str:
    return f"{_display_name(drug_id)} × {condition.strip()}"


def _classify_tier(chunks: list[dict]) -> PrecautionTier:
    if not chunks:
        return "grey"

    combined_text = " ".join(c.get("text", "") for c in chunks)
    if _red_re.search(combined_text):
        return "red"
    if _yellow_re.search(combined_text):
        return "yellow"
    return "green"


def _tier_message(tier: PrecautionTier, drug_id: str, condition: str) -> str:
    label = _check_label(drug_id, condition)
    if tier == "red":
        return f"Contraindication or warning language found in indexed documents for {label}."
    if tier == "yellow":
        return f"Precaution language found in indexed documents for {label} — review citations."
    if tier == "green":
        return f"No precaution language found in indexed documents for {label}."
    return f"Insufficient data in indexed documents to assess {label}."


def _build_summary(tier: PrecautionTier, chunks: list[dict], drug_id: str, condition: str) -> str:
    if tier == "grey":
        return _tier_message(tier, drug_id, condition)

    if tier == "green":
        return (
            f"No significant precaution language found in indexed documents for "
            f"{_check_label(drug_id, condition)}. This does not confirm safety — "
            "consult source labels."
        )

    best = chunks[0]
    excerpt = _format_excerpt(best.get("text", ""), max_len=280)
    if excerpt:
        return excerpt
    return _tier_message(tier, drug_id, condition)


def _chunk_mentions_drug(chunk: dict, drug_id: str) -> bool:
    drugs = chunk.get("drugs")
    if isinstance(drugs, list) and drug_id in drugs:
        return True
    return drug_id in find_drugs_in_text(chunk.get("text", ""))


def _safe_retrieve(label: str, fn) -> list[dict]:
    try:
        return fn()
    except Exception as exc:
        print(f"  [drug-precautions] {label} skipped: {exc}")
        return []


def _retrieve_drug_chunks(drug_id: str, condition: str) -> list[dict]:
    """Hybrid retrieval for drug × patient context."""
    name = _display_name(drug_id)
    condition = condition.strip()

    metadata_filtered = _safe_retrieve(
        "metadata filter",
        lambda: retrieve_filtered(
            f"{name} {condition} contraindication precaution warning pregnancy",
            metadata_filter={"drugs": {"$in": [drug_id]}},
            top_k=4,
        ),
    )

    semantic = _safe_retrieve(
        "semantic search",
        lambda: retrieve(
            f"{name} drug {condition} contraindication precaution warning side effects",
            top_k=6,
        ),
    )
    semantic = [c for c in semantic if _chunk_mentions_drug(c, drug_id)]

    entry = get_drug(drug_id) or {}
    source_pdf = entry.get("source_pdf", "")
    per_label: list[dict] = []
    if source_pdf:
        per_label = _safe_retrieve(
            "label PDF",
            lambda: retrieve_for_source(
                f"{name} {condition} contraindication precaution warning pregnancy",
                source_pdf=source_pdf,
                top_k=4,
            ),
        )

    return merge_chunks(metadata_filtered, semantic, per_label)


def screen_drug(drug_id: str, condition: str) -> dict:
    if not condition or not condition.strip():
        raise ValueError("Condition is required for drug precaution screening")

    chunks = _retrieve_drug_chunks(drug_id, condition)
    tier = _classify_tier(chunks)

    return {
        "drug_id": drug_id,
        "condition": condition.strip(),
        "label": _check_label(drug_id, condition),
        "tier": tier,
        "summary": _build_summary(tier, chunks, drug_id, condition),
        "citations": chunks_to_citations(chunks),
    }


def screen_drug_precautions(drug_ids: list[str], condition: str) -> dict:
    """Screen each drug against the given patient context."""
    if not condition or not condition.strip():
        raise ValueError("Condition is required for drug precaution screening")

    unique: list[str] = []
    seen: set[str] = set()
    for drug_id in drug_ids:
        normalized = drug_id.lower().strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)

    checks = [screen_drug(d, condition) for d in unique]

    return {
        "checks": checks,
        "disclaimer": DISCLAIMER,
        "drugs_screened": unique,
        "condition": condition.strip(),
    }
