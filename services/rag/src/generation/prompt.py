"""
prompt.py — Assemble the RAG prompt from retrieved chunks.

WHY THIS IS THE MOST IMPORTANT FILE IN THE PROJECT:
  This is where RAG actually happens. The prompt controls everything:
  - Whether Gemini answers from your documents or from its training data
  - Whether citations are returned correctly
  - Whether hallucinations are prevented

THE CORE RAG INSTRUCTION:
  "Answer ONLY using the provided documents. If the answer is not in
   the documents, say so. Do not use outside knowledge."

  Without this, Gemini will confidently answer from its training data
  and the whole point of RAG is lost. With this, every answer is
  grounded in your indexed PDFs.

WHAT THE ASSEMBLED PROMPT LOOKS LIKE:
  System:
    You are MediQuery, a medical knowledge assistant...
    Answer ONLY from the provided document excerpts.
    ...

  Context documents:
    [Document 1 — Metformin.pdf, Page 4]
    "Metformin is first-line therapy for type 2 diabetes..."

    [Document 2 — WHO Diabetes treatment guidelines PDF.pdf, Page 12]
    "GI symptoms are most common in first weeks of treatment..."

  Question: What are the side effects of Metformin?

  Answer:
"""

import re

# ---------------------------------------------------------------------------
# System prompt — instructs Gemini how to behave
# ---------------------------------------------------------------------------
REFUSAL_PHRASE = "I could not find information about this in the indexed documents."

SYSTEM_PROMPT = """You are MediQuery, an internal medical knowledge assistant.
Answer using ONLY the document excerpts below. Synthesize across excerpts when helpful.

CRITICAL — when to answer vs refuse:
- ALWAYS answer if excerpts mention the drug(s), condition(s), or closely related concepts in the question.
- For contraindication, interaction, or "can a patient take X" questions: summarize warnings, precautions, renal/hepatic cautions, and monitoring guidance from the excerpts — even if the exact term (e.g. "renal artery stenosis") is not verbatim. Related terms (renal impairment, hypotension, kidney function) count.
- NEVER respond with "I could not find information about this in the indexed documents." when excerpts discuss the drug or a related condition. Use that phrase ONLY when excerpts are about unrelated drugs or conditions entirely.
- If excerpts partially address the question, answer with what they support and briefly note what is not covered in the documents.
- Do not open with "I could not find information" if the excerpts contain relevant warnings or precautions — lead with what the documents say.

Formatting:
- Be precise and clinical. Keep answers concise — 2-5 sentences unless more detail is needed.
- Cite document name and page when stating facts.
- Do not invent specific doses or clinical facts that are absent from the excerpts.

You are not a replacement for professional medical advice. You are a document
retrieval assistant helping clinical teams find information faster."""

RETRY_INSTRUCTION = """
IMPORTANT: Do NOT refuse. The excerpts above are relevant to the question.
Summarize what they say about the drugs/conditions asked about, including warnings and precautions.
Do NOT use the phrase "I could not find information about this in the indexed documents."
"""


def build_prompt(question: str, chunks: list[dict], *, retry: bool = False) -> list[dict]:
    """
    Build the message list for the Gemini API call.

    WHY A MESSAGE LIST:
      Gemini (like most LLMs) takes a structured list of messages rather than
      a raw string. We send two messages:
        1. A "user" message containing the system instructions + context + question
        (Gemini's chat API handles system prompts differently from OpenAI —
         we fold the system prompt into the first user message for simplicity)

    Args:
        question: The user's plain-English question.
        chunks:   Retrieved chunks from search.retrieve() — list of
                  { text, source, page, chunk_index, score }

    Returns:
        List of message dicts ready for the Gemini API:
        [{ "role": "user", "parts": ["...full prompt..."] }]
    """
    if not chunks:
        # No relevant chunks found — still ask Gemini but it should say so
        context_block = "No relevant document excerpts were found for this question."
    else:
        context_block = _build_context_block(chunks)

    full_prompt = f"""{SYSTEM_PROMPT}

---
DOCUMENT EXCERPTS:
{context_block}
---

QUESTION: {question}
{RETRY_INSTRUCTION if retry else ""}
ANSWER:"""

    return [
        {"role": "user", "parts": [full_prompt]}
    ]


def is_refusal_answer(answer: str) -> bool:
    lower = answer.lower()
    if REFUSAL_PHRASE.lower() in lower:
        return True
    # Hedged refusals with no substantive follow-through
    if "i could not find information" in lower and "however" not in lower:
        return True
    return False


_HEDGED_PREFIX = re.compile(
    r"^I could not find information about[^.]*\.\s*However,?\s*",
    re.IGNORECASE,
)


def clean_hedged_answer(answer: str) -> str:
    """
    Strip leading 'I could not find... However,' when the model hedges then answers.
    """
    cleaned = _HEDGED_PREFIX.sub("", answer.strip())
    if cleaned == answer.strip():
        return answer
    if cleaned:
        return cleaned[0].upper() + cleaned[1:]
    return answer


def _build_context_block(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a readable context block.

    Each chunk is labeled with its source and page so Gemini
    can reference them accurately in its answer.

    Example output:
      [Document 1 — Metformin.pdf, Page 4]
      Metformin is first-line therapy for type 2 diabetes mellitus...

      [Document 2 — WHO Diabetes treatment guidelines PDF.pdf, Page 12]
      GI symptoms are most common in the first weeks of treatment...
    """
    lines = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "?")
        page_label = int(page) if isinstance(page, float) and page == int(page) else page
        text = chunk.get("text", "").strip()

        lines.append(f"[Document {i} — {source}, Page {page_label}]")
        lines.append(text)
        lines.append("")  # blank line between chunks

    return "\n".join(lines).strip()


EXCERPT_MAX_CHARS = 400


def _format_excerpt(text: str, max_len: int = EXCERPT_MAX_CHARS) -> str:
    """Build a readable citation preview from a chunk (may start mid-paragraph)."""
    text = " ".join(text.split())
    if not text:
        return ""

    if len(text) <= max_len:
        return text

    excerpt = text[:max_len]
    last_sentence = excerpt.rfind(". ")
    if last_sentence > max_len * 0.4:
        excerpt = excerpt[: last_sentence + 1]
    else:
        excerpt = excerpt.rstrip() + "..."

    # Chunk overlap can leave text starting mid-sentence
    if text[0].islower() or (len(text) > 1 and text[0].isdigit()):
        excerpt = "..." + excerpt

    return excerpt


def extract_citations(chunks: list[dict]) -> list[dict]:
    """
    Build the citations list returned to the frontend.

    This is separate from the prompt — it's the structured data
    the React UI uses to render citation cards below each answer.

    Returns:
        [
          {
            "source":  "Metformin.pdf",
            "page":    4,
            "excerpt": "Metformin is first-line therapy...",
            "score":   0.91
          },
          ...
        ]
    """
    citations = []
    seen = set()  # deduplicate same source+page

    for chunk in chunks:
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", 0)
        key = f"{source}::{page}"

        if key in seen:
            continue
        seen.add(key)

        text = chunk.get("text", "")
        excerpt = _format_excerpt(text)

        citations.append({
            "source":  source,
            "page":    int(page) if isinstance(page, float) and page == int(page) else page,
            "excerpt": excerpt,
            "score":   chunk.get("score", 0),
        })

    return citations


if __name__ == "__main__":
    """Smoke test — print what the assembled prompt looks like."""
    test_chunks = [
        {
            "text":   "Metformin is first-line therapy for type 2 diabetes mellitus. It reduces hepatic glucose production and improves insulin sensitivity.",
            "source": "Metformin.pdf",
            "page":   4,
            "score":  0.91,
        },
        {
            "text":   "Common adverse effects of metformin include nausea, vomiting, diarrhea, and abdominal discomfort, especially during initiation.",
            "source": "Metformin.pdf",
            "page":   7,
            "score":  0.87,
        },
    ]

    messages = build_prompt("What are the side effects of Metformin?", test_chunks)
    print("=== ASSEMBLED PROMPT ===\n")
    print(messages[0]["parts"][0])

    print("\n=== CITATIONS ===\n")
    import json
    citations = extract_citations(test_chunks)
    print(json.dumps(citations, indent=2))