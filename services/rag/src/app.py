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
import threading
from pathlib import Path

# Make sure Python can find sibling modules
sys.path.insert(0, str(Path(__file__).parent))

import fitz  # PyMuPDF — validate uploads
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from retrieval.search import retrieve
from ingestion.pipeline import ingest_pdf, IngestError
from ingestion.uploader import delete_vectors_by_source
from ingestion.job_store import (
    create_job,
    complete_job,
    fail_job,
    get_job,
    make_progress_callback,
)
from generation.prompt import (
    build_prompt,
    clean_hedged_answer,
    extract_citations,
    is_refusal_answer,
)
from generation.llm import generate_answer
from generation.confidence import compute_confidence
from drugs.detection import detect_drugs_from_conversation
from drugs.interactions import screen_drugs, DISCLAIMER as INTERACTION_DISCLAIMER
from drugs.precautions import screen_drug_precautions
from drugs.registry import get_profile, list_profiles, drugs_in_knowledge_base, normalize_drug_id
from vaccines.detection import detect_vaccines_from_conversation
from vaccines.precautions import screen_vaccines, DISCLAIMER as VACCINE_DISCLAIMER
from vaccines.registry import (
    get_profile as get_vaccine_profile,
    list_profiles as list_vaccine_profiles,
    vaccines_in_knowledge_base,
    normalize_vaccine_id,
)
from documents.catalog import (
    build_documents_payload,
    count_pdfs,
    destination_for_upload,
    find_pdf_by_path,
    load_sections,
    resolve_documents_root,
)

_SRC_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB — large clinical PDFs (200–300 pages)


def _count_indexed_documents() -> int:
    return count_pdfs()


def _resolve_documents_dir() -> Path:
    return resolve_documents_root()


load_dotenv(_SRC_DIR.parent.parent.parent / ".env")

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
        {
          "question": "Is that safe for elderly patients?",
          "history": [
            {
              "question": "What are the side effects of Metformin?",
              "answer": "Metformin commonly causes..."
            }
          ]
        }

        history is optional — up to 3 prior Q&A pairs for follow-up context.
        Pinecone search still uses only the current question.

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

    conversation_history = []
    raw_history = data.get("history") or []
    if isinstance(raw_history, list):
        for turn in raw_history[-3:]:
            if not isinstance(turn, dict):
                continue
            q = str(turn.get("question", "")).strip()
            a = str(turn.get("answer", "")).strip()
            if q and a:
                conversation_history.append({"question": q, "answer": a})

    print(f"\n[/query] Question: {question}")
    if conversation_history:
        print(f"[/query] History: {len(conversation_history)} turn(s)")

    try:
        docs_dir = _resolve_documents_dir()
        drug_context = detect_drugs_from_conversation(
            question,
            conversation_history,
            docs_dir=docs_dir,
        )
        vaccine_context = detect_vaccines_from_conversation(
            question,
            conversation_history,
            docs_dir=docs_dir,
        )

        # Step 1 — retrieve relevant chunks from Pinecone
        chunks = retrieve(question)

        total_documents = _count_indexed_documents()

        if not chunks:
            return jsonify({
                "answer": "I could not find any relevant information in the indexed documents.",
                "citations": [],
                "confidence": compute_confidence([], total_documents),
                "detected_drugs": drug_context["detected_drugs"],
                "knowledge_base_drugs": drug_context["knowledge_base_drugs"],
                "drug_profiles": drug_context["profiles"],
                "detected_vaccines": vaccine_context["detected_vaccines"],
                "knowledge_base_vaccines": vaccine_context["knowledge_base_vaccines"],
                "vaccine_profiles": vaccine_context["profiles"],
            })

        # Step 2 — build RAG prompt (history helps resolve follow-ups like "that")
        messages = build_prompt(question, chunks, conversation_history=conversation_history)

        # Step 3 — call Gemini (retry once if it refuses despite relevant chunks)
        answer = generate_answer(messages)
        if chunks and is_refusal_answer(answer):
            print("[/query] Model refused despite retrieved chunks — retrying with stricter prompt")
            messages = build_prompt(
                question, chunks, retry=True, conversation_history=conversation_history
            )
            answer = generate_answer(messages)

        # Step 4 — assemble citations + confidence for frontend
        citations = extract_citations(chunks)
        confidence = compute_confidence(chunks, total_documents)
        answer = clean_hedged_answer(answer)

        print(
            f"[/query] Done — {len(citations)} citations, "
            f"confidence={confidence['tier']} ({confidence['score_percent']}%)"
        )

        return jsonify({
            "answer":     answer,
            "citations":  citations,
            "confidence": confidence,
            "detected_drugs": drug_context["detected_drugs"],
            "knowledge_base_drugs": drug_context["knowledge_base_drugs"],
            "drug_profiles": drug_context["profiles"],
            "detected_vaccines": vaccine_context["detected_vaccines"],
            "knowledge_base_vaccines": vaccine_context["knowledge_base_vaccines"],
            "vaccine_profiles": vaccine_context["profiles"],
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
    Return documents grouped by section for the sidebar.
    Reads from documents/<section>/*.pdf — same layout as ingestion.
    """
    payload = build_documents_payload()
    payload["section_options"] = load_sections()
    return jsonify(payload)


# ---------------------------------------------------------------------------
# Drug interaction screening
# ---------------------------------------------------------------------------
@app.route("/interactions", methods=["POST"])
def interactions():
    """
    Screen drug pairs from indexed documents (on-demand from UI).

    Request body:
        { "drugs": ["metformin", "lisinopril"], "condition": "optional" }

    Requires at least 2 drug ids from the registry.
    """
    data = request.get_json(silent=True) or {}
    raw_drugs = data.get("drugs") or []

    if not isinstance(raw_drugs, list) or len(raw_drugs) < 2:
        return jsonify({
            "error": "Request must include 'drugs' with at least 2 drug names",
        }), 400

    docs_dir = _resolve_documents_dir()
    kb_ids = set(drugs_in_knowledge_base(docs_dir))
    normalized: list[str] = []
    for name in raw_drugs:
        drug_id = normalize_drug_id(str(name))
        if not drug_id:
            return jsonify({"error": f"Unknown drug: {name}"}), 400
        if drug_id not in kb_ids:
            return jsonify({
                "error": f"Drug '{drug_id}' is not in the indexed knowledge base",
            }), 400
        if drug_id not in normalized:
            normalized.append(drug_id)

    if len(normalized) < 2:
        return jsonify({
            "error": "At least 2 distinct drugs are required for interaction screening",
        }), 400

    condition = data.get("condition")
    condition_str = str(condition).strip() if condition else None

    print(f"\n[/interactions] Screening: {normalized}")

    try:
        result = screen_drugs(normalized, condition=condition_str)
        return jsonify(result)
    except Exception as exc:
        print(f"[/interactions] Error: {exc}")
        return jsonify({"error": "Interaction screening failed"}), 500


@app.route("/drugs", methods=["GET"])
def drugs_list():
    """Return drug profiles for drugs in the knowledge base."""
    docs_dir = _resolve_documents_dir()
    kb_ids = drugs_in_knowledge_base(docs_dir)
    return jsonify({
        "drugs": list_profiles(kb_ids),
        "disclaimer": INTERACTION_DISCLAIMER,
    })


@app.route("/drugs/<drug_id>", methods=["GET"])
def drug_detail(drug_id: str):
    """Return a single drug profile by registry id."""
    normalized = normalize_drug_id(drug_id) or drug_id.lower()
    profile = get_profile(normalized)
    if not profile:
        return jsonify({"error": "Drug not found in registry"}), 404
    return jsonify({"drug": profile})


# ---------------------------------------------------------------------------
# Drug precaution screening (drug × patient context)
# ---------------------------------------------------------------------------
@app.route("/drug-precautions", methods=["POST"])
def drug_precautions():
    """
    Screen drugs against patient context from indexed label documents.

    Request body:
        { "drugs": ["metformin"], "condition": "conditions: Pregnancy" }

    Requires at least 1 drug id and a non-empty condition.
    """
    data = request.get_json(silent=True) or {}
    raw_drugs = data.get("drugs") or []
    condition = data.get("condition")

    if not isinstance(raw_drugs, list) or len(raw_drugs) < 1:
        return jsonify({
            "error": "Request must include 'drugs' with at least 1 drug name",
        }), 400

    condition_str = str(condition).strip() if condition else ""
    if not condition_str:
        return jsonify({
            "error": "Request must include a non-empty 'condition' (patient context)",
        }), 400

    docs_dir = _resolve_documents_dir()
    kb_ids = set(drugs_in_knowledge_base(docs_dir))
    normalized: list[str] = []
    for name in raw_drugs:
        drug_id = normalize_drug_id(str(name))
        if not drug_id:
            return jsonify({"error": f"Unknown drug: {name}"}), 400
        if drug_id not in kb_ids:
            return jsonify({
                "error": f"Drug '{drug_id}' is not in the indexed knowledge base",
            }), 400
        if drug_id not in normalized:
            normalized.append(drug_id)

    print(f"\n[/drug-precautions] Screening: {normalized} | Condition: {condition_str}")

    try:
        result = screen_drug_precautions(normalized, condition_str)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        print(f"[/drug-precautions] Error: {exc}")
        return jsonify({"error": "Drug precaution screening failed"}), 500


# ---------------------------------------------------------------------------
# Vaccine precaution screening
# ---------------------------------------------------------------------------
@app.route("/vaccine-precautions", methods=["POST"])
def vaccine_precautions():
    """
    Screen vaccines against a patient condition from indexed schedule documents.

    Request body:
        { "vaccines": ["influenza"], "condition": "pregnancy" }

    Requires at least 1 vaccine id and a non-empty condition.
    """
    data = request.get_json(silent=True) or {}
    raw_vaccines = data.get("vaccines") or []
    condition = data.get("condition")

    if not isinstance(raw_vaccines, list) or len(raw_vaccines) < 1:
        return jsonify({
            "error": "Request must include 'vaccines' with at least 1 vaccine name",
        }), 400

    condition_str = str(condition).strip() if condition else ""
    if not condition_str:
        return jsonify({
            "error": "Request must include a non-empty 'condition' (patient context)",
        }), 400

    docs_dir = _resolve_documents_dir()
    kb_ids = set(vaccines_in_knowledge_base(docs_dir))
    normalized: list[str] = []
    for name in raw_vaccines:
        vaccine_id = normalize_vaccine_id(str(name))
        if not vaccine_id:
            return jsonify({"error": f"Unknown vaccine: {name}"}), 400
        if vaccine_id not in kb_ids:
            return jsonify({
                "error": f"Vaccine '{vaccine_id}' is not in the indexed knowledge base",
            }), 400
        if vaccine_id not in normalized:
            normalized.append(vaccine_id)

    print(f"\n[/vaccine-precautions] Screening: {normalized} | Condition: {condition_str}")

    try:
        result = screen_vaccines(normalized, condition_str)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        print(f"[/vaccine-precautions] Error: {exc}")
        return jsonify({"error": "Vaccine precaution screening failed"}), 500


@app.route("/vaccines", methods=["GET"])
def vaccines_list():
    """Return vaccine profiles for vaccines in the knowledge base."""
    docs_dir = _resolve_documents_dir()
    kb_ids = vaccines_in_knowledge_base(docs_dir)
    return jsonify({
        "vaccines": list_vaccine_profiles(kb_ids),
        "disclaimer": VACCINE_DISCLAIMER,
    })


@app.route("/vaccines/<vaccine_id>", methods=["GET"])
def vaccine_detail(vaccine_id: str):
    """Return a single vaccine profile by registry id."""
    normalized = normalize_vaccine_id(vaccine_id) or vaccine_id.lower()
    profile = get_vaccine_profile(normalized)
    if not profile:
        return jsonify({"error": "Vaccine not found in registry"}), 404
    return jsonify({"vaccine": profile})


def _sanitize_filename(raw: str) -> str | None:
    """Strip path components; return None if not a .pdf filename."""
    name = os.path.basename(raw or "").strip()
    if not name or not name.lower().endswith(".pdf"):
        return None
    return name


def _validate_pdf_file(path: Path) -> str | None:
    """Return an error message if the file is not a readable PDF."""
    try:
        with fitz.open(str(path)) as doc:
            if len(doc) == 0:
                return "File could not be opened as a PDF"
    except Exception:
        return "File could not be opened as a PDF"
    return None


# ---------------------------------------------------------------------------
# Upload endpoint — save PDF + run ingestion pipeline
# ---------------------------------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload():
    """
    Receive a PDF from Express, save to documents/<section>/, index into Pinecone.

    multipart/form-data fields: file, section (section id from sections.json)
    """
    if "file" not in request.files:
        return jsonify({"error": "Request must include a 'file' field"}), 400

    upload_file = request.files["file"]
    if not upload_file or not upload_file.filename:
        return jsonify({"error": "No file provided"}), 400

    if request.content_length and request.content_length > MAX_UPLOAD_BYTES:
        return jsonify({"error": "File exceeds 50MB limit"}), 413

    filename = _sanitize_filename(upload_file.filename)
    if not filename:
        return jsonify({"error": "Only PDF files are supported"}), 400

    section_id = request.form.get("section", "").strip()
    dest, rel_path = destination_for_upload(filename, section_id)
    if dest is None:
        return jsonify({"error": rel_path}), 400

    reindexed = dest.exists()

    try:
        upload_file.save(str(dest))
    except OSError as exc:
        print(f"[/upload] Save failed: {exc}")
        return jsonify({"error": "Failed to save uploaded file"}), 500

    if dest.stat().st_size > MAX_UPLOAD_BYTES:
        dest.unlink(missing_ok=True)
        return jsonify({"error": "File exceeds 50MB limit"}), 413

    pdf_error = _validate_pdf_file(dest)
    if pdf_error:
        dest.unlink(missing_ok=True)
        return jsonify({"error": pdf_error}), 400

    job_id = create_job(_resolve_documents_dir(), rel_path, reindexed=reindexed)
    print(f"\n[/upload] Job {job_id}: {rel_path}" + (" (re-index)" if reindexed else ""))

    def run_ingestion() -> None:
        on_progress = make_progress_callback(_resolve_documents_dir(), job_id)
        try:
            result = ingest_pdf(
                dest,
                reindexed=reindexed,
                on_progress=on_progress,
                source_key=rel_path,
            )
            complete_job(_resolve_documents_dir(), job_id, result)
        except IngestError as exc:
            print(f"[/upload] Ingest error: {exc.message}")
            job = get_job(_resolve_documents_dir(), job_id)
            fail_job(
                _resolve_documents_dir(),
                job_id,
                exc.message,
                stage=job.get("stage") if job else None,
            )
        except Exception as exc:
            dest.unlink(missing_ok=True)
            print(f"[/upload] Unexpected error: {exc}")
            fail_job(_resolve_documents_dir(), job_id, "Indexing failed — file was not added")

    threading.Thread(target=run_ingestion, daemon=True).start()

    return jsonify({
        "job_id": job_id,
        "filename": filename,
        "path": rel_path,
        "section": section_id,
        "status": "accepted",
        "reindexed": reindexed,
    }), 202


@app.route("/upload/status/<job_id>", methods=["GET"])
def upload_status(job_id: str):
    """Poll ingestion progress for an upload job."""
    docs_dir = _resolve_documents_dir()
    job = get_job(docs_dir, job_id)
    if not job:
        return jsonify({"error": "Upload job not found"}), 404
    return jsonify(job)


# ---------------------------------------------------------------------------
# Delete endpoint — remove PDF + Pinecone vectors
# ---------------------------------------------------------------------------
@app.route("/delete/<path:filename>", methods=["DELETE"])
def delete_document(filename: str):
    """Delete a document from disk and remove its vectors from Pinecone."""
    clean_path = filename.replace("\\", "/").strip().lstrip("/")
    if ".." in clean_path.split("/"):
        return jsonify({"error": "Invalid document path"}), 400

    dest = find_pdf_by_path(clean_path)
    if not dest:
        return jsonify({"error": "Document not found"}), 404

    print(f"\n[/delete] Removing: {clean_path}")

    try:
        vectors_deleted = delete_vectors_by_source(clean_path)
        dest.unlink()
        return jsonify({
            "status": "deleted",
            "filename": dest.name,
            "path": clean_path,
            "vectors_deleted": vectors_deleted,
        })
    except Exception as exc:
        print(f"[/delete] Error: {exc}")
        return jsonify({"error": "Failed to delete document"}), 500


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
    print(f"   POST /interactions")
    print(f"   GET  /drugs")
    print(f"   GET  /drugs/<id>")
    print(f"   POST /upload")
    print(f"   GET  /upload/status/<job_id>")
    print(f"   DELETE /delete/<filename>")
    print(f" Debug mode: {debug}\n")

    app.run(host="0.0.0.0", port=port, debug=debug)