"""
patient_context.py — Parse and apply session patient context to RAG queries.

Used lightly: optional prompt tailoring + gated retrieval augmentation.
"""

from __future__ import annotations

import re

_SAFETY_PATTERN = re.compile(
    r"\b("
    r"safe|safety|contraindicat|precaution|caution|warning|"
    r"interact|interaction|allerg|pregnancy|pregnant|"
    r"dose|dosing|eligib|can\s+(i|they|we|a\s+patient)\s+take|"
    r"should\s+(i|they|we)\s+(take|use|get|receive)|"
    r"renal|kidney|hepatic|liver|immunocompromis|"
    r"side\s+effect|adverse"
    r")\b",
    re.IGNORECASE,
)


def parse_patient_context(raw: object | None) -> dict[str, str]:
    """Normalize patient context from API request body."""
    empty = {"age": "", "gender": "", "allergies": "", "conditions": ""}
    if not isinstance(raw, dict):
        return empty

    return {
        "age": str(raw.get("age", "") or "").strip(),
        "gender": str(raw.get("gender", "") or "").strip(),
        "allergies": str(raw.get("allergies", "") or "").strip(),
        "conditions": str(raw.get("conditions", "") or "").strip(),
    }


def has_patient_context(ctx: dict[str, str]) -> bool:
    return bool(build_context_string(ctx))


def build_context_string(ctx: dict[str, str]) -> str:
    parts: list[str] = []

    age = ctx.get("age", "").strip()
    if age:
        parts.append(age if age.lower().startswith("age") else f"age {age}")

    gender = ctx.get("gender", "").strip()
    if gender:
        parts.append(gender)

    allergies = ctx.get("allergies", "").strip()
    if allergies:
        parts.append(f"allergies: {allergies}")

    conditions = ctx.get("conditions", "").strip()
    if conditions:
        parts.append(f"conditions: {conditions}")

    return "; ".join(parts)


def format_context_lines(ctx: dict[str, str]) -> list[str]:
    lines: list[str] = []
    if ctx.get("age", "").strip():
        lines.append(f"Age: {ctx['age'].strip()}")
    if ctx.get("gender", "").strip():
        lines.append(f"Gender: {ctx['gender'].strip()}")
    if ctx.get("allergies", "").strip():
        lines.append(f"Allergies: {ctx['allergies'].strip()}")
    if ctx.get("conditions", "").strip():
        lines.append(f"Conditions: {ctx['conditions'].strip()}")
    return lines


def should_augment_retrieval(
    question: str,
    *,
    patient_ctx: dict[str, str],
    detected_drugs: list[str],
    detected_vaccines: list[str],
) -> bool:
    """
    Augment Pinecone retrieval only when context is set and the question
    is likely about safety, eligibility, or clinical entities in context.
    """
    if not has_patient_context(patient_ctx):
        return False

    if detected_drugs or detected_vaccines:
        return True

    return bool(_SAFETY_PATTERN.search(question or ""))


def build_prompt_context_block(ctx: dict[str, str]) -> str:
    """Soft instruction block for build_prompt — only when context is set."""
    lines = format_context_lines(ctx)
    if not lines:
        return ""

    bullet_lines = "\n".join(f"- {line}" for line in lines)
    return f"""
SESSION PATIENT CONTEXT (tailor only when excerpts support it):
{bullet_lines}

If the question involves safety, precautions, contraindications, interactions, allergies,
pregnancy, dosing, or eligibility, briefly note any patient-specific cautions found in
the excerpts. If excerpts do not address this profile, answer normally and do not speculate."""
