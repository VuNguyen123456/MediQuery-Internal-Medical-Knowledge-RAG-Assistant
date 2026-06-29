"""
format_answer.py — Post-process LLM answers into scannable markdown sections.
"""

from __future__ import annotations

import re

_BULLET = re.compile(r"^[-*•]\s+", re.MULTILINE)
_SENTENCE = re.compile(r"[^.!?]+[.!?]+(?:\s|$)")
_CITE = re.compile(r"(\[[^[\]]+?,\s*Page\s+[^\]]+\])\s*$")
_CITE_INLINE = re.compile(r"\[[^[\]]+?,\s*Page\s+[^\]]+\]")
_LEGACY_DOC_CITE = re.compile(r"\[Document\s+(\d+),\s*Page\s+([^\]]+)\]", re.IGNORECASE)
_CITE_PLACEHOLDER = re.compile(r"__CITE_(\d+)__")

_SECTION_HEADINGS = {
    "common": "Common effects",
    "serious": "Serious warning — lactic acidosis",
    "risks": "Risk factors",
    "renal": "Renal & elderly patients",
    "action": "What to do",
    "other": "Additional notes",
}

_SECTION_ORDER = ["common", "serious", "risks", "renal", "action", "other"]


def _classify_sentence(sentence: str) -> str:
    lower = sentence.lower()
    if re.search(r"lactic acidosis|hemodialysis|can be fatal|anion gap|lactate:pyruvate", lower):
        return "serious"
    if re.search(r"discontinu|educated about|report symptoms|healthcare provider|families should", lower):
        return "action"
    if re.search(r"renal|kidney|elderly|clearance|half-life|excreted by the kidney", lower):
        return "renal"
    if re.search(
        r"higher risk|risk of|alcohol|dehydration|liver problems|severe infection|stroke|surgery|heart attack",
        lower,
    ):
        return "risks"
    if re.search(r"diarrhea|nausea|upset stomach|metallic taste|hypoglyce|gi symptom", lower):
        return "common"
    return "other"


def _mask_citations(text: str) -> tuple[str, list[str]]:
    """Protect [File.pdf, Page N] from sentence splitting on dots in filenames."""
    cites: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        cites.append(match.group(0))
        return f"__CITE_{len(cites) - 1}__"

    return _CITE_INLINE.sub(_stash, text), cites


def _unmask_citations(text: str, cites: list[str]) -> str:
    def _restore(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        return cites[idx] if 0 <= idx < len(cites) else match.group(0)

    return _CITE_PLACEHOLDER.sub(_restore, text)


def _extract_sentences(content: str) -> list[str]:
    protected, cites = _mask_citations(content)
    flat = " ".join(_BULLET.sub("", line.strip()) for line in protected.splitlines())
    flat = re.sub(r"\s+", " ", flat).strip()
    matches = _SENTENCE.findall(flat)
    sentences = [_unmask_citations(s.strip(), cites) for s in matches if s.strip()]
    return sentences or [content.strip()]


def _explode_clauses(sentence: str) -> list[str]:
    cite_match = _CITE.search(sentence)
    cite = cite_match.group(1) if cite_match else ""
    body = sentence[: cite_match.start()].strip() if cite_match else sentence

    parts = [p.strip() for p in body.split(";") if p.strip()]
    if len(parts) <= 1:
        return [sentence]

    result: list[str] = []
    for i, part in enumerate(parts):
        clause = part if re.search(r"[.!?]$", part) else f"{part}."
        if i == len(parts) - 1 and cite:
            clause = f"{clause} {cite}"
        result.append(clause)
    return result


def _is_well_structured(content: str) -> bool:
    headings = re.findall(r"\*\*[^*]+\*\*", content)
    bullets = [line for line in content.splitlines() if _BULLET.match(line.strip())]
    short_bullets = [b for b in bullets if len(_BULLET.sub("", b.strip())) <= 160]
    return len(headings) >= 2 and len(short_bullets) >= 2


def needs_structure(content: str) -> bool:
    """Only restructure dense prose — never compress already-sectioned answers."""
    trimmed = content.strip()
    if len(trimmed) < 400:
        return False
    if _is_well_structured(trimmed):
        return False
    if re.search(r"\*\*[^*]+\*\*", trimmed):
        return False

    bullets = [line for line in trimmed.splitlines() if _BULLET.match(line.strip())]
    if bullets and all(len(line) > 140 for line in bullets):
        return True

    return "\n\n" not in trimmed and len(trimmed) > 500


def structure_answer(content: str) -> str:
    trimmed = content.strip()
    if not trimmed or not needs_structure(trimmed):
        return trimmed

    sentences = _extract_sentences(trimmed)
    if len(sentences) <= 2:
        return trimmed

    intro_count = 1 if len(sentences) <= 5 else 2
    intro = " ".join(sentences[:intro_count])
    body = sentences[intro_count:]

    buckets: dict[str, list[str]] = {}
    for sentence in body:
        key = _classify_sentence(sentence)
        buckets.setdefault(key, []).append(sentence)

    parts = [intro]
    for key in _SECTION_ORDER:
        items = buckets.get(key)
        if not items:
            continue

        heading = _SECTION_HEADINGS[key]
        parts.append("")
        if key == "serious":
            parts.append(f">>> {heading}")
        else:
            parts.append(f"**{heading}**")

        for item in items:
            for clause in _explode_clauses(item):
                parts.append(f"- {clause}")

    return "\n\n".join(parts).strip()


def rewrite_document_citations(answer: str, chunks: list[dict]) -> str:
    """Replace legacy [Document N, Page P] with [Filename.pdf, Page P]."""
    if not chunks or "[Document " not in answer:
        return answer

    from generation.prompt import format_citation_label

    index_map = {
        i: format_citation_label(chunk.get("source", "Unknown"), chunk.get("page", "?"))
        for i, chunk in enumerate(chunks, 1)
    }

    def _replace(match: re.Match[str]) -> str:
        doc_num = int(match.group(1))
        return index_map.get(doc_num, match.group(0))

    return _LEGACY_DOC_CITE.sub(_replace, answer)


def _best_citation_for_text(text: str, chunks: list[dict]) -> str:
    from generation.prompt import format_citation_label

    words = set(re.findall(r"\w{4,}", text.lower()))
    best = chunks[0]
    best_score = -1
    for chunk in chunks:
        chunk_words = set(re.findall(r"\w{4,}", chunk.get("text", "").lower()))
        score = len(words & chunk_words)
        if score > best_score:
            best_score = score
            best = chunk
    return format_citation_label(best.get("source", "Unknown"), best.get("page", "?"))


def enrich_bullet_citations(answer: str, chunks: list[dict]) -> str:
    """Append source citations to bullets when the model omitted them."""
    if not chunks:
        return answer

    lines: list[str] = []
    for line in answer.split("\n"):
        stripped = line.strip()
        if _BULLET.match(stripped) and not _CITE_INLINE.search(stripped):
            cite = _best_citation_for_text(stripped, chunks)
            lines.append(f"{line.rstrip()} {cite}")
        else:
            lines.append(line)
    return "\n".join(lines)


def format_answer_for_display(answer: str, chunks: list[dict] | None = None) -> str:
    """Apply citation rewrite, enrichment, and light structuring when needed."""
    if chunks:
        answer = rewrite_document_citations(answer, chunks)
        answer = enrich_bullet_citations(answer, chunks)
    return structure_answer(answer)
