"""
detection.py — Detect drugs in user questions and conversation history.
"""

from __future__ import annotations

from drugs.registry import drugs_in_knowledge_base, find_drugs_in_text, get_profile


def detect_drugs_from_conversation(
    question: str,
    history: list[dict] | None = None,
    *,
    docs_dir=None,
) -> dict:
    """
    Detect drugs in the current question and recent history.

    Returns:
        detected_drugs: drugs mentioned in question + history
        question_drugs: drugs in the current question only
        knowledge_base_drugs: all registered drugs present in documents/
        profiles: mini profiles for detected drugs
    """
    kb_drug_ids = drugs_in_knowledge_base(docs_dir)

    parts = [question or ""]
    if history:
        for turn in history:
            if isinstance(turn, dict):
                parts.append(str(turn.get("question", "")))
                parts.append(str(turn.get("answer", "")))

    combined_text = "\n".join(parts)
    detected = find_drugs_in_text(combined_text, allowed_ids=kb_drug_ids)
    question_drugs = find_drugs_in_text(question or "", allowed_ids=kb_drug_ids)

    profiles = [get_profile(d) for d in detected]
    profiles = [p for p in profiles if p]

    return {
        "detected_drugs": detected,
        "question_drugs": question_drugs,
        "knowledge_base_drugs": kb_drug_ids,
        "profiles": profiles,
    }
