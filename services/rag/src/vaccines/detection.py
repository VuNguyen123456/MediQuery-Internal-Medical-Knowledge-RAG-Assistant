"""
detection.py — Detect vaccines in user questions and conversation history.
"""

from __future__ import annotations

from vaccines.registry import (
    find_vaccines_in_text,
    get_profile,
    vaccines_in_knowledge_base,
)


def detect_vaccines_from_conversation(
    question: str,
    history: list[dict] | None = None,
    *,
    docs_dir=None,
) -> dict:
    """
    Detect vaccines in the current question and recent history.

    Returns:
        detected_vaccines: vaccines mentioned in question + history
        question_vaccines: vaccines in the current question only
        knowledge_base_vaccines: all registered vaccines present in documents/
        profiles: mini profiles for detected vaccines
    """
    kb_vaccine_ids = vaccines_in_knowledge_base(docs_dir)

    parts = [question or ""]
    if history:
        for turn in history:
            if isinstance(turn, dict):
                parts.append(str(turn.get("question", "")))
                parts.append(str(turn.get("answer", "")))

    combined_text = "\n".join(parts)
    detected = find_vaccines_in_text(combined_text, allowed_ids=kb_vaccine_ids)
    question_vaccines = find_vaccines_in_text(question or "", allowed_ids=kb_vaccine_ids)

    profiles = [get_profile(v) for v in detected]
    profiles = [p for p in profiles if p]

    return {
        "detected_vaccines": detected,
        "question_vaccines": question_vaccines,
        "knowledge_base_vaccines": kb_vaccine_ids,
        "profiles": profiles,
    }
