"""
search.py — Embed a question and retrieve the most relevant chunks from Pinecone.

WHY THIS EXISTS:
  Pinecone stores 535 vectors. When a user asks a question, we need to find
  which chunks are most semantically similar to that question.

  We use the SAME embedding model as ingestion (all-MiniLM-L6-v2).
  This is critical — if we embedded documents with model A and queries
  with model B, the vectors would live in different spaces and similarity
  scores would be meaningless.

HOW SIMILARITY WORKS:
  Both the question and every stored chunk are 384-dimensional vectors.
  Pinecone computes cosine similarity between the question vector and all
  535 stored vectors, then returns the top K most similar ones.

  Cosine similarity = 1.0 means identical meaning, 0.0 means unrelated.
  In practice, a good match scores 0.7+, weak match 0.4-0.6.

WHAT COMES BACK:
  [
    {
      "text":        "Metformin is first-line therapy...",
      "source":      "Metformin.pdf",
      "page":        4,
      "chunk_index": 12,
      "score":       0.91
    },
    ...
  ]
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # must match ingestion
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mediquery")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TOP_K = 4  # number of chunks to retrieve per query

# ---------------------------------------------------------------------------
# Singletons — loaded once per process, reused across all queries
# ---------------------------------------------------------------------------
_embedding_model: SentenceTransformer | None = None
_pinecone_index = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        print("  [search] Loading embedding model (cached after first load)...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index

    if not PINECONE_API_KEY:
        raise EnvironmentError("PINECONE_API_KEY not set in .env")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    _pinecone_index = pc.Index(INDEX_NAME)
    return _pinecone_index


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """
    Convert a question to a vector and retrieve the most relevant chunks.

    Args:
        question: The user's plain-English question.
        top_k:    Number of chunks to return (default 4).

    Returns:
        List of chunk dicts sorted by relevance (highest score first):
        [{ text, source, page, chunk_index, score }, ...]

    Raises:
        EnvironmentError: If PINECONE_API_KEY is not set.
        Exception: If Pinecone query fails.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    # Step 1 — embed the question using the same model as ingestion
    model = _get_embedding_model()
    question_vector = model.encode(question, convert_to_numpy=True).tolist()

    # Step 2 — query Pinecone for most similar chunks
    index = _get_pinecone_index()
    results = index.query(
        vector=question_vector,
        top_k=top_k,
        include_metadata=True,   # we need text, source, page back
    )

    # Step 3 — extract and return clean chunk dicts
    chunks = []
    for match in results.matches:
        metadata = match.metadata or {}
        chunks.append({
            "text":        metadata.get("text", ""),
            "source":      metadata.get("source", "unknown"),
            "page":        metadata.get("page", 0),
            "chunk_index": metadata.get("chunk_index", 0),
            "score":       round(float(match.score), 4),
        })

    print(f"  [search] '{question[:60]}...' → {len(chunks)} chunks retrieved")
    for c in chunks:
        print(f"    score={c['score']} | {c['source']} p.{c['page']} | {c['text'][:60]}...")

    return chunks


if __name__ == "__main__":
    """Quick smoke test — query against your real Pinecone index."""
    test_questions = [
        "What are the side effects of Metformin?",
        "Who should get the flu vaccine?",
        "What is the first-line treatment for hypertension?",
    ]

    for q in test_questions:
        print(f"\nQuestion: {q}")
        print("-" * 60)
        results = retrieve(q)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] score={r['score']} | {r['source']} p.{r['page']}")
            print(f"       {r['text'][:120]}...")
        print()