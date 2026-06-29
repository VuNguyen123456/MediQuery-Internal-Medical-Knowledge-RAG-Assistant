"""
uploader.py — Embed chunks and upsert them into Pinecone.

TWO THINGS HAPPEN HERE:

1. EMBEDDING (HuggingFace all-MiniLM-L6-v2)
   Each chunk's text → 384-dimensional float vector.
   "Metformin causes nausea" and "GI side effects of Metformin" will have
   very similar vectors even though they share no words — that's semantic search.
   The model runs locally, no API key needed.

2. PINECONE UPSERT
   Each vector is stored with its metadata (text, source, page, chunk_index).
   Upsert = insert if new, overwrite if same ID already exists.
   This makes re-running ingestion safe — no duplicates.

BATCHING:
   We upsert in batches of 100. Pinecone recommends this for performance.
   A 200-page PDF ≈ 400 chunks ≈ 4 batches.
"""

import os
import time
from pathlib import Path
from typing import Callable, Generator
from dotenv import load_dotenv

# Pinecone v3+ API
from pinecone import Pinecone, ServerlessSpec

# sentence-transformers runs the embedding model locally
from sentence_transformers import SentenceTransformer

from drugs.tagging import tag_chunk_drugs
from vaccines.tagging import tag_chunk_vaccines

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality
EMBEDDING_DIM = 384                      # must match index dimension
UPSERT_BATCH_SIZE = 100                  # Pinecone recommended batch size
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mediquery")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# ---------------------------------------------------------------------------
# Module-level singletons — loaded once, reused across all documents
# ---------------------------------------------------------------------------
_embedding_model: SentenceTransformer | None = None
_pinecone_index = None


def _get_embedding_model() -> SentenceTransformer:
    """Lazy-load the embedding model (downloaded once, cached locally)."""
    global _embedding_model
    if _embedding_model is None:
        print(f"  [uploader] Loading embedding model '{EMBEDDING_MODEL}'...")
        print(f"  [uploader] (First run: downloads ~90MB model — subsequent runs use cache)")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"  [uploader] Model loaded OK")
    return _embedding_model


def _get_pinecone_index():
    """Lazy-init Pinecone client and ensure the index exists."""
    global _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index

    if not PINECONE_API_KEY:
        raise EnvironmentError(
            "PINECONE_API_KEY not set. Add it to your .env file."
        )

    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if it doesn't exist
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"  [uploader] Index '{INDEX_NAME}' not found — creating it...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",            # cosine similarity for semantic search
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )
        # Wait for index to be ready
        print(f"  [uploader] Waiting for index to initialize...")
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
        print(f"  [uploader] Index '{INDEX_NAME}' ready OK")
    else:
        print(f"  [uploader] Using existing index '{INDEX_NAME}' OK")

    _pinecone_index = pc.Index(INDEX_NAME)
    return _pinecone_index


def embed_chunks(
    chunks: list[dict],
    *,
    on_batch_progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """
    Add an 'embedding' field to each chunk dict.

    Args:
        chunks: Output from chunker.chunk_pages()
        on_batch_progress: Optional callback(current, total) after each encode batch.

    Returns:
        Same list with 'embedding' added to each item:
        [{ ..., "embedding": [0.23, -0.87, 0.45, ...] }, ...]
    """
    model = _get_embedding_model()
    texts = [c["text"] for c in chunks]
    encode_batch_size = 32
    total = len(chunks)

    print(f"  [uploader] Embedding {total} chunks...")

    all_embeddings = []
    for start in range(0, total, encode_batch_size):
        batch_texts = texts[start : start + encode_batch_size]
        batch_vectors = model.encode(
            batch_texts,
            batch_size=encode_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        all_embeddings.extend(batch_vectors)
        if on_batch_progress:
            on_batch_progress(min(start + len(batch_texts), total), total)

    for chunk, vector in zip(chunks, all_embeddings):
        chunk["embedding"] = vector.tolist()

    print(f"  [uploader] Embedding complete OK ({EMBEDDING_DIM}-dim vectors)")
    return chunks


def upsert_to_pinecone(
    chunks: list[dict],
    *,
    on_batch_progress: Callable[[int, int], None] | None = None,
) -> int:
    """
    Upsert embedded chunks into Pinecone.

    Args:
        chunks: Chunks with 'embedding' field (output of embed_chunks)
        on_batch_progress: Optional callback(current, total) after each upsert batch.

    Returns:
        Total number of vectors upserted.
    """
    index = _get_pinecone_index()

    total_upserted = 0
    batches = list(_batch(chunks, UPSERT_BATCH_SIZE))
    total = len(chunks)

    print(f"  [uploader] Upserting {total} vectors in {len(batches)} batches...")

    for batch_num, batch in enumerate(batches, start=1):
        vectors = []
        for chunk in batch:
            metadata = {
                "text":        chunk["text"],
                "source":      chunk["source"],
                "source_basename": Path(chunk["source"]).name,
                "page":        chunk["page"],
                "chunk_index": chunk["chunk_index"],
            }
            drug_meta = tag_chunk_drugs(chunk["text"])
            if drug_meta.get("drugs"):
                metadata["drugs"] = drug_meta["drugs"]
            if drug_meta.get("drug_pair"):
                metadata["drug_pair"] = drug_meta["drug_pair"]

            vaccine_meta = tag_chunk_vaccines(chunk["text"])
            if vaccine_meta.get("vaccines"):
                metadata["vaccines"] = vaccine_meta["vaccines"]

            vectors.append({
                "id": chunk["chunk_id"],
                "values": chunk["embedding"],
                "metadata": metadata,
            })

        index.upsert(vectors=vectors)
        total_upserted += len(vectors)
        if on_batch_progress:
            on_batch_progress(total_upserted, total)
        print(f"  [uploader] Batch {batch_num}/{len(batches)} — {total_upserted} vectors upserted so far")

    print(f"  [uploader] Upsert complete OK - {total_upserted} total vectors in Pinecone")
    return total_upserted


def delete_vectors_by_source(source: str) -> int:
    """
    Delete Pinecone vectors for a document (relative path and legacy basename).
    """
    index = _get_pinecone_index()
    basename = Path(source.replace("\\", "/")).name
    total = 0

    for key in {source, basename}:
        if not key:
            continue
        response = index.delete(filter={"source": {"$eq": key}})
        deleted = getattr(response, "deleted_count", None)
        if deleted is None and isinstance(response, dict):
            deleted = response.get("deleted_count", 0)
        total += int(deleted or 0)

    if basename and basename != source:
        response = index.delete(filter={"source_basename": {"$eq": basename}})
        deleted = getattr(response, "deleted_count", None)
        if deleted is None and isinstance(response, dict):
            deleted = response.get("deleted_count", 0)
        total += int(deleted or 0)

    print(f"  [uploader] Deleted {total} vectors for source '{source}'")
    return total


def _batch(items: list, size: int) -> Generator:
    """Yield successive fixed-size batches from a list."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


if __name__ == "__main__":
    """Smoke test — runs the full embed + upsert on dummy data."""
    print("Testing uploader with 3 dummy chunks...\n")

    test_chunks = [
        {
            "chunk_id":    "test_p0001_c0000",
            "text":        "Metformin is first-line therapy for type 2 diabetes mellitus.",
            "source":      "test.pdf",
            "page":        1,
            "chunk_index": 0,
        },
        {
            "chunk_id":    "test_p0001_c0001",
            "text":        "Common side effects include nausea, vomiting, and diarrhea.",
            "source":      "test.pdf",
            "page":        1,
            "chunk_index": 1,
        },
        {
            "chunk_id":    "test_p0002_c0002",
            "text":        "Metformin is contraindicated in patients with severe kidney disease.",
            "source":      "test.pdf",
            "page":        2,
            "chunk_index": 2,
        },
    ]

    embedded = embed_chunks(test_chunks)
    print(f"\nVector preview: {embedded[0]['embedding'][:5]}... (length {len(embedded[0]['embedding'])})\n")

    upserted = upsert_to_pinecone(embedded)
    print(f"\nDone — {upserted} vectors in Pinecone.")