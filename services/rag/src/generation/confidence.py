"""
confidence.py — Grade retrieval quality into four tiers for the frontend.

Tiers (based on best chunk cosine similarity score):
  high      >= 0.75  green  — no banner, citation indicators only
  moderate  >= 0.55  yellow
  low       >= 0.40  orange
  very_low  <  0.40  red
"""

from __future__ import annotations

THRESHOLD_HIGH = 0.75
THRESHOLD_MODERATE = 0.55
THRESHOLD_LOW = 0.40
STRONG_MATCH_SCORE = 0.75

TIER_CONFIG = {
    "high": {
        "label": "High Confidence",
        "message": "Answer drawn from closely matched source material.",
        "show_banner": False,
        "show_rephrase_hint": False,
    },
    "moderate": {
        "label": "Moderate Confidence",
        "message": (
            "Partial match — answer is based on related but not exact source material. "
            "Review citations."
        ),
        "show_banner": True,
        "show_rephrase_hint": False,
    },
    "low": {
        "label": "Low Confidence",
        "message": (
            "Weak match — source material may not fully address this question. "
            "Answer could be incomplete or imprecise."
        ),
        "show_banner": True,
        "show_rephrase_hint": True,
    },
    "very_low": {
        "label": "Very Low / No Match",
        "message": (
            "No strong match found in indexed documents. This answer should not be "
            "relied upon — consult source documents directly."
        ),
        "show_banner": True,
        "show_rephrase_hint": True,
    },
}

REPHRASE_HINT = (
    "Try rephrasing your question using terminology from the source documents."
)


def score_to_tier(score: float) -> str:
    if score >= THRESHOLD_HIGH:
        return "high"
    if score >= THRESHOLD_MODERATE:
        return "moderate"
    if score >= THRESHOLD_LOW:
        return "low"
    return "very_low"


def _format_page(page) -> int | float | str:
    if isinstance(page, float) and page == int(page):
        return int(page)
    return page


def _spread_summary(chunks: list[dict]) -> str:
    strong = sum(1 for c in chunks if c.get("score", 0) >= STRONG_MATCH_SCORE)
    weak = len(chunks) - strong

    if strong and weak:
        strong_noun = "match" if strong == 1 else "matches"
        weak_noun = "match" if weak == 1 else "matches"
        return f"{strong} strong {strong_noun}, {weak} weak {weak_noun}"

    if strong:
        noun = "match" if strong == 1 else "matches"
        return f"{strong} strong {noun} found"

    if weak:
        noun = "match" if weak == 1 else "matches"
        return f"{weak} weak {noun}"

    return "No matches retrieved"


def compute_confidence(chunks: list[dict], total_documents: int) -> dict:
    """
    Build the confidence object returned alongside answer + citations.

    Uses the highest retrieval score to pick the overall tier. Spread and
    document coverage are derived from all returned chunks.
    """
    if not chunks:
        config = TIER_CONFIG["very_low"]
        return {
            "tier": "very_low",
            "score": 0.0,
            "score_percent": 0,
            "label": config["label"],
            "message": config["message"],
            "show_banner": config["show_banner"],
            "show_rephrase_hint": config["show_rephrase_hint"],
            "rephrase_hint": REPHRASE_HINT,
            "best_match": {"source": "—", "page": "—"},
            "total_documents": total_documents,
            "documents_contributing": 0,
            "search_summary": f"0 of {total_documents} documents searched",
            "coverage_summary": f"Answer pulled from 0 of {total_documents} indexed documents",
            "spread_summary": "No matches retrieved",
        }

    best = chunks[0]
    best_score = float(best.get("score", 0))
    tier = score_to_tier(best_score)
    config = TIER_CONFIG[tier]

    contributing_sources = {c.get("source", "unknown") for c in chunks}
    documents_contributing = len(contributing_sources)
    total = max(total_documents, 1)

    best_page = _format_page(best.get("page", "?"))
    best_source = best.get("source", "Unknown")

    return {
        "tier": tier,
        "score": round(best_score, 4),
        "score_percent": round(best_score * 100),
        "label": config["label"],
        "message": config["message"],
        "show_banner": config["show_banner"],
        "show_rephrase_hint": config["show_rephrase_hint"],
        "rephrase_hint": REPHRASE_HINT,
        "best_match": {
            "source": best_source,
            "page": best_page,
        },
        "total_documents": total_documents,
        "documents_contributing": documents_contributing,
        "search_summary": (
            f"{documents_contributing} of {total} documents searched"
        ),
        "coverage_summary": (
            f"Answer pulled from {documents_contributing} of {total} indexed documents"
        ),
        "spread_summary": _spread_summary(chunks),
    }
