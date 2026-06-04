"""
app.py — Flask entry point for the MediQuery RAG service.

This is the only file Express talks to. One endpoint:
  POST /query   → runs the full RAG pipeline, returns answer + citations

INTERNAL ONLY:
  This service runs on port 5000 inside Docker.
  It is never directly accessible from the browser.
  Express (port 8000) proxies all requests to it.
  This means API keys (Pinecone, Gemini) never reach the frontend.

PIPELINE ON EVERY REQUEST:
  1. Receive { question } from Express
  2. search.retrieve()  → embed question, find top 4 chunks in Pinecone
  3. prompt.build_prompt() → assemble RAG prompt with chunks as context
  4. llm.generate_answer() → call Gemini, get grounded answer
  5. prompt.extract_citations() → build citation list for frontend
  6. Return { answer, citations } to Express
"""

import sys
import os
from pathlib import Path

# Make sure Python can find sibling modules
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from retrieval.search import retrieve
from generation.prompt import (
    build_prompt,
    clean_hedged_answer,
    extract_citations,
    is_refusal_answer,
)
from generation.llm import generate_answer

load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")

app = Flask(__name__)
CORS(app)  # Express is on a different port — CORS needed for local dev


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    """Express pings this to confirm Flask is running before proxying."""
    return jsonify({"status": "ok", "service": "mediquery-rag"})


# ---------------------------------------------------------------------------
# Core RAG endpoint
# ---------------------------------------------------------------------------
@app.route("/query", methods=["POST"])
def query():
    """
    Run the full RAG pipeline for a user question.

    Request body:
        { "question": "What are the side effects of Metformin?" }

    Response:
        {
          "answer": "Metformin commonly causes gastrointestinal side effects...",
          "citations": [
            {
              "source":  "Metformin.pdf",
              "page":    4,
              "excerpt": "Common adverse effects include nausea, vomiting...",
              "score":   0.91
            },
            ...
          ]
        }

    Error responses:
        400 — missing or empty question
        500 — pipeline failure (Pinecone/Gemini error)
    """
    # --- Validate input ---
    data = request.get_json(silent=True)
    if not data or not data.get("question"):
        return jsonify({"error": "Request body must include a 'question' field"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    print(f"\n[/query] Question: {question}")

    try:
        # Step 1 — retrieve relevant chunks from Pinecone
        chunks = retrieve(question)

        if not chunks:
            return jsonify({
                "answer": "I could not find any relevant information in the indexed documents.",
                "citations": []
            })

        # Step 2 — build RAG prompt
        messages = build_prompt(question, chunks)

        # Step 3 — call Gemini (retry once if it refuses despite relevant chunks)
        answer = generate_answer(messages)
        if chunks and is_refusal_answer(answer):
            print("[/query] Model refused despite retrieved chunks — retrying with stricter prompt")
            messages = build_prompt(question, chunks, retry=True)
            answer = generate_answer(messages)

        # Step 4 — assemble citations for frontend
        citations = extract_citations(chunks)
        answer = clean_hedged_answer(answer)

        print(f"[/query] Done — {len(citations)} citations")

        return jsonify({
            "answer":    answer,
            "citations": citations,
        })

    except EnvironmentError as e:
        # Missing API key — config issue
        print(f"[/query] Config error: {e}")
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        # Unexpected failure — log and return generic error
        print(f"[/query] Unexpected error: {e}")
        return jsonify({"error": "An error occurred processing your question"}), 500


# ---------------------------------------------------------------------------
# Documents list endpoint (for Ring 3 sidebar)
# ---------------------------------------------------------------------------
@app.route("/documents", methods=["GET"])
def documents():
    """
    Return a list of indexed document names.
    Used by the React frontend sidebar to show what's in the knowledge base.
    Reads from the /documents folder — same source of truth as ingestion.
    """
    docs_dir = Path(__file__).resolve().parent.parent.parent.parent / "documents"

    if not docs_dir.exists():
        return jsonify({"documents": []})

    pdf_files = [
        {"name": f.name, "size_kb": round(f.stat().st_size / 1024, 1)}
        for f in sorted(docs_dir.glob("*.pdf"))
    ]

    return jsonify({"documents": pdf_files})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    print(f"\n MediQuery RAG Service starting on port {port}")
    print(f" Endpoints:")
    print(f"   GET  /health")
    print(f"   POST /query")
    print(f"   GET  /documents")
    print(f" Debug mode: {debug}\n")

    app.run(host="0.0.0.0", port=port, debug=debug)