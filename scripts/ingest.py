"""
ingest.py — Full ingestion pipeline orchestrator.

Run this once per document (or re-run anytime — it's idempotent).

USAGE:
  # Ingest all PDFs in the documents/ folder:
  python scripts/ingest.py

  # Ingest a specific PDF:
  python scripts/ingest.py documents/NIH_Diabetes_Guidelines.pdf

WHAT IT DOES:
  For each PDF:
    1. Parse  → extract text + page numbers
    2. Chunk  → split into 500-word overlapping chunks
    3. Embed  → convert each chunk to a 384-dim vector (HuggingFace)
    4. Upsert → store vectors + metadata in Pinecone

  After this runs, you can verify in the Pinecone dashboard:
    → Go to https://app.pinecone.io
    → Select your index ("mediquery")
    → You should see vectors with metadata like source, page, text

IDEMPOTENCY:
  chunk_ids are deterministic (based on filename + page + position).
  Pinecone upsert overwrites on matching ID — no duplicates on re-run.
"""

import sys
import os
import time
from pathlib import Path

# Windows consoles often default to cp1252; allow Unicode status symbols in logs
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (OSError, ValueError):
        pass

# Make sure Python can find ingestion modules under services/rag/src
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "services/rag/src"))

from ingestion.parser import parse_pdf
from ingestion.chunker import chunk_pages
from ingestion.uploader import embed_chunks, upsert_to_pinecone


DOCUMENTS_DIR = _PROJECT_ROOT / "documents"


def ingest_file(pdf_path: Path) -> dict:
    """
    Run the full ingestion pipeline for a single PDF.

    Returns a summary dict for reporting.
    """
    print(f"\n{'='*60}")
    print(f"  Ingesting: {pdf_path.name}")
    print(f"{'='*60}")

    start = time.time()

    # Step 1 — Parse
    print(f"\n[1/3] Parsing PDF...")
    pages = parse_pdf(str(pdf_path))

    if not pages:
        print(f"  WARNING: No text extracted from {pdf_path.name} - skipping.")
        print(f"  (Is this a scanned/image PDF? OCR not supported in Ring 1.)")
        return {"file": pdf_path.name, "status": "skipped", "reason": "no text"}

    # Step 2 — Chunk
    print(f"\n[2/3] Chunking text...")
    chunks = chunk_pages(pages, pdf_path.name)

    if not chunks:
        print(f"  WARNING: No chunks produced - skipping.")
        return {"file": pdf_path.name, "status": "skipped", "reason": "no chunks"}

    # Step 3 — Embed + Upsert
    print(f"\n[3/3] Embedding and uploading to Pinecone...")
    embedded_chunks = embed_chunks(chunks)
    total_vectors = upsert_to_pinecone(embedded_chunks)

    elapsed = time.time() - start

    summary = {
        "file":     pdf_path.name,
        "status":   "success",
        "pages":    len(pages),
        "chunks":   len(chunks),
        "vectors":  total_vectors,
        "elapsed":  f"{elapsed:.1f}s",
    }

    print(f"\n  Done in {elapsed:.1f}s")
    print(f"    Pages parsed:   {len(pages)}")
    print(f"    Chunks created: {len(chunks)}")
    print(f"    Vectors stored: {total_vectors}")

    return summary


def main():
    print("\n" + "="*60)
    print("  MediQuery - Document Ingestion Pipeline")
    print("="*60)

    # Determine which files to ingest
    if len(sys.argv) > 1:
        # Specific files passed as arguments
        pdf_paths = [Path(p) for p in sys.argv[1:]]
        for p in pdf_paths:
            if not p.exists():
                print(f"ERROR: File not found: {p}")
                sys.exit(1)
            if p.suffix.lower() != ".pdf":
                print(f"ERROR: Not a PDF: {p}")
                sys.exit(1)
    else:
        # Default: all PDFs in documents/
        if not DOCUMENTS_DIR.exists():
            print(f"ERROR: Documents folder not found: {DOCUMENTS_DIR}")
            print(f"  Create the folder and add your medical PDFs:")
            print(f"  mkdir documents/")
            print(f"  # then add NIH, WHO, FDA PDFs")
            sys.exit(1)

        pdf_paths = sorted(DOCUMENTS_DIR.glob("*.pdf"))

        if not pdf_paths:
            print(f"\nNo PDFs found in {DOCUMENTS_DIR}/")
            print(f"\nDownload some free medical documents:")
            print(f"  NIH:  https://www.nhlbi.nih.gov/health-topics/guidelines")
            print(f"  FDA:  https://www.accessdata.fda.gov/drugsatfda_docs/label/")
            print(f"  WHO:  https://www.who.int/publications/")
            print(f"  CDC:  https://www.cdc.gov/vaccines/hcp/")
            sys.exit(0)

    print(f"\nFound {len(pdf_paths)} PDF(s) to ingest:")
    for p in pdf_paths:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  - {p.name} ({size_mb:.1f} MB)")

    # Run pipeline
    results = []
    for pdf_path in pdf_paths:
        result = ingest_file(pdf_path)
        results.append(result)

    # Final summary
    print(f"\n\n{'='*60}")
    print(f"  INGESTION COMPLETE - Summary")
    print(f"{'='*60}")

    total_vectors = 0
    for r in results:
        status_icon = "OK" if r["status"] == "success" else "SKIP"
        if r["status"] == "success":
            print(f"  [{status_icon}] {r['file']}")
            print(f"      {r['pages']} pages -> {r['chunks']} chunks -> {r['vectors']} vectors ({r['elapsed']})")
            total_vectors += r["vectors"]
        else:
            print(f"  [{status_icon}] {r['file']} - SKIPPED ({r.get('reason', 'unknown')})")

    print(f"\n  Total vectors in Pinecone: ~{total_vectors} (+ any previously indexed)")
    print(f"\n  Verify at: https://app.pinecone.io -> index '{os.getenv('PINECONE_INDEX_NAME', 'mediquery')}'")
    print(f"  Next: build the Flask query endpoint (Ring 2).\n")


if __name__ == "__main__":
    main()