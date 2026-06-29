"""
pipeline.py — Shared ingestion pipeline for CLI and Flask upload endpoint.

parse_pdf → chunk_pages → embed_chunks → upsert_to_pinecone
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from documents.catalog import relative_pdf_path, resolve_documents_root
from ingestion.parser import parse_pdf
from ingestion.chunker import chunk_pages
from ingestion.uploader import embed_chunks, upsert_to_pinecone

REINDEX_WARNING = (
    "This document is already indexed — it will be re-indexed."
)
IMAGE_ONLY_ERROR = (
    "This PDF appears to be image-only. Text extraction is not supported."
)

ProgressCallback = Callable[
    [str, str | None, int | None, int | None], None
]


class IngestError(Exception):
    """Raised when ingestion fails with a user-facing message."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def ingest_pdf(
    filepath: Path,
    *,
    reindexed: bool = False,
    on_progress: ProgressCallback | None = None,
    source_key: str | None = None,
) -> dict:
    """
    Run the full ingestion pipeline for a single PDF on disk.

    Args:
        filepath: Absolute path to the saved PDF.
        reindexed: True if this file replaced an existing document.
        on_progress: Optional callback(stage, message, current, total).

    Returns:
        Result dict with filename, pages, chunks, vectors, status, etc.

    Raises:
        IngestError: On empty text, no chunks, or Pinecone failure.
    """
    if not filepath.exists():
        raise IngestError(f"PDF not found: {filepath.name}", 404)

    filename = filepath.name
    docs_root = resolve_documents_root()
    if source_key is None:
        try:
            source_key = relative_pdf_path(filepath, docs_root)
        except ValueError:
            source_key = filename
    start = time.time()

    def report(
        stage: str,
        message: str | None = None,
        current: int | None = None,
        total: int | None = None,
    ) -> None:
        if on_progress:
            on_progress(stage, message, current, total)

    report("parsing", "Reading PDF pages...")
    pages = parse_pdf(str(filepath))
    if not pages:
        filepath.unlink(missing_ok=True)
        raise IngestError(IMAGE_ONLY_ERROR, 422)

    report("chunking", f"Splitting {len(pages)} pages into chunks...")
    chunks = chunk_pages(pages, source_key)
    if not chunks:
        filepath.unlink(missing_ok=True)
        raise IngestError(
            "No text chunks could be produced from this PDF.", 422
        )

    total_chunks = len(chunks)

    try:
        report("embedding", f"Embedding {total_chunks} chunks...", 0, total_chunks)

        def embed_progress(current: int, total: int) -> None:
            report(
                "embedding",
                f"Embedding chunks ({current}/{total})...",
                current,
                total,
            )

        embedded_chunks = embed_chunks(
            chunks,
            on_batch_progress=embed_progress,
        )

        report("upserting", f"Saving {total_chunks} vectors to Pinecone...", 0, total_chunks)

        def upsert_progress(current: int, total: int) -> None:
            report(
                "upserting",
                f"Uploading to Pinecone ({current}/{total})...",
                current,
                total,
            )

        total_vectors = upsert_to_pinecone(
            embedded_chunks,
            on_batch_progress=upsert_progress,
        )
    except Exception as exc:
        filepath.unlink(missing_ok=True)
        raise IngestError(
            f"Indexing failed — file was not added: {exc}", 500
        ) from exc

    elapsed = time.time() - start

    result = {
        "filename": filename,
        "path": source_key,
        "pages": len(pages),
        "chunks": len(chunks),
        "vectors": total_vectors,
        "status": "success",
        "reindexed": reindexed,
        "warning": REINDEX_WARNING if reindexed else None,
        "elapsed": f"{elapsed:.1f}s",
    }

    report("done", "Indexing complete")
    print(
        f"  [pipeline] {filename}: {result['pages']} pages → "
        f"{result['chunks']} chunks → {result['vectors']} vectors ({result['elapsed']})"
    )
    return result
