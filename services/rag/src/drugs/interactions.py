"""
interactions.py — Document-grounded drug interaction screening.

Hybrid retrieval per pair:
  1. Co-mention chunks (Pinecone metadata drug_pair)
  2. Semantic search for explicit interaction language
  3. Per-drug label chunks mentioning the other drug or interaction sections

Tiers (document-grounded only):
  red    — explicit warning language in retrieved excerpts
  yellow — co-mentioned without clear interaction language
  green  — interaction-focused search found no warnings in documents
  grey   — insufficient relevant excerpts in indexed sources
"""

from __future__ import annotations

import itertools
import re
from typing import Literal

from drugs.registry import get_drug
from generation.prompt import _format_excerpt
from retrieval.search import retrieve, retrieve_filtered, retrieve_for_source
from screening.common import chunks_to_citations, merge_chunks

InteractionTier = Literal["red", "yellow", "green", "grey"]

DISCLAIMER = (
    "Interaction screening is based on indexed documents only — not a substitute "
    "for pharmacist review, drug interaction databases, or clinical judgment."
)

WARNING_PATTERNS = [
    r"\bcontraindicat",
    r"\bavoid\b",
    r"\bdo not use\b",
    r"\bdrug interaction",
    r"\binteraction[s]?\b",
    r"\bmonitor (closely|renal|kidney|potassium|blood pressure)",
    r"\bincreased risk\b",
    r"\blactic acidosis\b",
    r"\bhyperkalemia\b",
    r"\bhypotension\b",
    r"\brenal impairment\b",
    r"\bconcomitant use\b",
    r"\bcaution\b",
    r"\bwarning[s]?\b",
    r"\bprecaution[s]?\b",
]

CO_MENTION_ONLY_PATTERNS = [
    r"\bconcomitant\b",
    r"\bconcurrent\b",
    r"\btogether\b",
    r"\bcombined\b",
    r"\bwith other\b",
]

_warning_re = re.compile("|".join(WARNING_PATTERNS), re.IGNORECASE)
_comention_re = re.compile("|".join(CO_MENTION_ONLY_PATTERNS), re.IGNORECASE)


def _display_name(drug_id: str) -> str:
    entry = get_drug(drug_id)
    if entry:
        return entry.get("display_name") or entry.get("canonical_name") or drug_id.title()
    return drug_id.title()


def _pair_label(drug_a: str, drug_b: str) -> str:
    return f"{_display_name(drug_a)} × {_display_name(drug_b)}"


def _sorted_pair(drug_a: str, drug_b: str) -> tuple[str, str]:
    return tuple(sorted([drug_a, drug_b]))  # type: ignore[return-value]


def _classify_tier(chunks: list[dict], *, co_mention_chunks: list[dict]) -> InteractionTier:
    if not chunks:
        return "grey"

    combined_text = " ".join(c.get("text", "") for c in chunks)
    if _warning_re.search(combined_text):
        return "red"

    if co_mention_chunks:
        co_text = " ".join(c.get("text", "") for c in co_mention_chunks)
        if _comention_re.search(co_text) or len(co_mention_chunks) > 0:
            return "yellow"

    return "green"


def _tier_message(tier: InteractionTier, drug_a: str, drug_b: str) -> str:
    label = _pair_label(drug_a, drug_b)
    if tier == "red":
        return f"Warning language found in indexed documents for {label}."
    if tier == "yellow":
        return f"{label} mentioned together in documents — interaction context unclear. Review citations."
    if tier == "green":
        return f"No interaction warning found in indexed documents for {label}."
    return f"Insufficient data in indexed documents to assess {label}."


def _build_summary(tier: InteractionTier, chunks: list[dict], drug_a: str, drug_b: str) -> str:
    if tier == "grey":
        return _tier_message(tier, drug_a, drug_b)

    if tier == "green":
        return (
            f"No significant interaction warning found in indexed documents for "
            f"{_pair_label(drug_a, drug_b)}. This does not confirm safety — consult source labels."
        )

    best = chunks[0]
    excerpt = _format_excerpt(best.get("text", ""), max_len=280)
    if excerpt:
        return excerpt
    return _tier_message(tier, drug_a, drug_b)


def _retrieve_pair_chunks(
    drug_a: str,
    drug_b: str,
    *,
    condition: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return (all_relevant_chunks, co_mention_chunks)."""
    a, b = _sorted_pair(drug_a, drug_b)
    pair_key = f"{a}|{b}"

    name_a = _display_name(a)
    name_b = _display_name(b)
    condition_suffix = f" {condition.strip()}" if condition and condition.strip() else ""

    co_mention = retrieve_filtered(
        f"{name_a} {name_b} interaction concomitant{condition_suffix}",
        metadata_filter={"drug_pair": {"$eq": pair_key}},
        top_k=4,
    )

    semantic = retrieve(
        f"{name_a} {name_b} drug interaction concomitant concurrent use"
        f"{condition_suffix} contraindication",
        top_k=4,
    )
    semantic = [
        c for c in semantic
        if a in (c.get("drugs") or find_drugs_in_chunk(c)) or b in (c.get("drugs") or find_drugs_in_chunk(c))
    ]

    entry_a = get_drug(a) or {}
    entry_b = get_drug(b) or {}
    source_a = entry_a.get("source_pdf", "")
    source_b = entry_b.get("source_pdf", "")

    per_a = retrieve_for_source(
        f"drug interactions {name_b} concomitant medications warnings{condition_suffix}",
        source_pdf=source_a,
        top_k=3,
    ) if source_a else []

    per_b = retrieve_for_source(
        f"drug interactions {name_a} concomitant medications warnings{condition_suffix}",
        source_pdf=source_b,
        top_k=3,
    ) if source_b else []

    all_chunks = merge_chunks(co_mention, semantic, per_a, per_b)
    return all_chunks, co_mention


def find_drugs_in_chunk(chunk: dict) -> list[str]:
    """Read drug ids from chunk metadata or re-detect from text."""
    drugs = chunk.get("drugs")
    if isinstance(drugs, list) and drugs:
        return drugs
    from drugs.registry import find_drugs_in_text
    return find_drugs_in_text(chunk.get("text", ""))


def screen_pair(drug_a: str, drug_b: str, *, condition: str | None = None) -> dict:
    if drug_a == drug_b:
        raise ValueError("Cannot screen interaction for the same drug twice")

    chunks, co_mention = _retrieve_pair_chunks(drug_a, drug_b, condition=condition)
    tier = _classify_tier(chunks, co_mention_chunks=co_mention)

    return {
        "drug_a": drug_a,
        "drug_b": drug_b,
        "label": _pair_label(drug_a, drug_b),
        "tier": tier,
        "summary": _build_summary(tier, chunks, drug_a, drug_b),
        "citations": chunks_to_citations(chunks),
    }


def screen_drugs(drug_ids: list[str], *, condition: str | None = None) -> dict:
    """
    Screen all unique pairs among detected drugs.

    When condition is provided, it is included in document retrieval queries.
    """
    unique = []
    seen: set[str] = set()
    for drug_id in drug_ids:
        normalized = drug_id.lower().strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)

    condition_str = condition.strip() if condition and condition.strip() else None
    pairs = list(itertools.combinations(unique, 2))
    results = [screen_pair(a, b, condition=condition_str) for a, b in pairs]

    response: dict = {
        "pairs": results,
        "disclaimer": DISCLAIMER,
        "drugs_screened": unique,
    }

    if condition_str:
        response["condition_note"] = (
            f"Patient context '{condition_str}' was included in document search queries."
        )

    return response
