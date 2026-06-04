"""
llm.py — Send the assembled RAG prompt to Gemini and get an answer.

MODEL CHOICE — gemini-2.5-flash-lite:
  Free-tier friendly lite model. For RAG the LLM's job is simple:
  read a few short excerpts and summarize an answer.

TEMPERATURE = 0:
  Fully deterministic. Medical assistant needs consistency —
  same question should give the same answer every time.
"""

import os
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
TEMPERATURE = 0
MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# Singleton model
# ---------------------------------------------------------------------------
_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model

    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey"
        )

    genai.configure(api_key=GEMINI_API_KEY)
    _model = genai.GenerativeModel(MODEL_NAME)
    return _model


def generate_answer(messages: list[dict]) -> str:
    """
    Send the prompt to Gemini and return the answer text.

    Args:
        messages: Output from prompt.build_prompt() —
                  [{ "role": "user", "parts": ["...full prompt..."] }]

    Returns:
        The answer text string from Gemini.
    """
    model = _get_model()
    prompt = messages[0]["parts"][0]

    print(f"  [llm] Calling {MODEL_NAME}...")
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
        ),
    )

    answer = response.text.strip()
    print(f"  [llm] Response received ({len(answer)} chars)")
    return answer


if __name__ == "__main__":
    """Smoke test."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from generation.prompt import build_prompt

    test_chunks = [
        {
            "text":   "Metformin is first-line therapy for type 2 diabetes mellitus. It reduces hepatic glucose production and improves insulin sensitivity.",
            "source": "Metformin.pdf",
            "page":   4,
            "score":  0.91,
        },
        {
            "text":   "Common adverse effects of metformin include nausea, vomiting, diarrhea, and abdominal discomfort, especially during initiation of therapy.",
            "source": "Metformin.pdf",
            "page":   7,
            "score":  0.87,
        },
    ]

    messages = build_prompt("What are the side effects of Metformin?", test_chunks)
    answer = generate_answer(messages)
    print("\n=== GEMINI ANSWER ===\n")
    print(answer)
