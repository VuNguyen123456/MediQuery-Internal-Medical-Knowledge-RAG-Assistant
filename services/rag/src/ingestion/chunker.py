"""
chunker.py — Split PDF pages into overlapping chunks for vector indexing.

WHY CHUNKING EXISTS:
  You can't embed an entire 200-page PDF as one vector — it loses all granularity.
  You can't embed page-by-page either — a single page may be too long, and a
  concept might span a page boundary.

  We split into ~500-word chunks with 50-word overlap:
  - 500 words → enough context for Gemini to answer meaningfully
  - 50-word overlap → prevents losing context exactly at a chunk boundary
    (e.g., a sentence that starts at the end of chunk 3 and finishes in chunk 4)

WHAT A CHUNK LOOKS LIKE:
  {
    "chunk_id":    "NIH_Diabetes_Guidelines_p12_c003",
    "text":        "Metformin is first-line therapy for type 2 diabetes...",
    "source":      "NIH_Diabetes_Guidelines.pdf",
    "page":        12,
    "chunk_index": 3,
  }

  chunk_id is deterministic — same doc always produces same IDs.
  This means re-running ingestion won't duplicate vectors (Pinecone upserts).
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration — tune these if retrieval quality needs adjustment
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500        # target words per chunk
CHUNK_OVERLAP = 50      # words shared between consecutive chunks

# RecursiveCharacterTextSplitter works in characters, not words.
# Average English word ≈ 5 characters + 1 space = 6 chars.
CHARS_PER_WORD = 6
CHUNK_SIZE_CHARS = CHUNK_SIZE * CHARS_PER_WORD          # 3000
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP * CHARS_PER_WORD    # 300

# Split hierarchy: try paragraph → sentence → word → character
# This preserves semantic boundaries as much as possible
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_pages(pages: list[dict], source_filename: str) -> list[dict]:
    """
    Split a list of parsed pages into overlapping text chunks.

    Args:
        pages:           Output from parser.parse_pdf() — list of {page, text}
        source_filename: PDF filename (e.g. "NIH_Diabetes_Guidelines.pdf")
                         Stored in metadata so citations work correctly.

    Returns:
        List of chunk dicts ready for embedding + Pinecone upload:
        [{ chunk_id, text, source, page, chunk_index }, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=SEPARATORS,
        length_function=len,
    )

    # Normalize filename — strip path, keep just the filename
    source = Path(source_filename).name

    # Build a clean stem for chunk IDs (no spaces, no extension)
    source_stem = Path(source).stem.replace(" ", "_")

    all_chunks = []
    global_chunk_index = 0

    for page_data in pages:
        page_num = page_data["page"]
        page_text = page_data["text"]

        if not page_text.strip():
            continue

        # Split this page's text into chunks
        raw_chunks = splitter.split_text(page_text)

        for local_idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            # Deterministic ID: source_stem + page + global index
            chunk_id = f"{source_stem}_p{page_num:04d}_c{global_chunk_index:04d}"

            all_chunks.append({
                "chunk_id":    chunk_id,
                "text":        chunk_text,
                "source":      source,
                "page":        page_num,
                "chunk_index": global_chunk_index,
            })

            global_chunk_index += 1

    print(
        f"  [chunker] '{source}' -> {len(pages)} pages -> {len(all_chunks)} chunks "
        f"(~{CHUNK_SIZE} words each, {CHUNK_OVERLAP}-word overlap)"
    )
    return all_chunks


if __name__ == "__main__":
    """Smoke test — combine with parser output."""
    import sys
    from parser import parse_pdf

    if len(sys.argv) < 2:
        print("Usage: python chunker.py <path_to_pdf>")
        sys.exit(1)

    pages = parse_pdf(sys.argv[1])
    chunks = chunk_pages(pages, sys.argv[1])

    print(f"\nSample chunks (first 3):\n")
    for c in chunks[:3]:
        preview = c["text"][:200].replace("\n", " ")
        print(f"  [{c['chunk_id']}] page={c['page']}")
        print(f"  Text: {preview}...\n")