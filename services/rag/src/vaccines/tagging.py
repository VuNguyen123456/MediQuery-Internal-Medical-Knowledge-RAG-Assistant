"""
tagging.py — Tag chunks with vaccine metadata during ingestion.
"""

from __future__ import annotations

from vaccines.registry import find_vaccines_in_text


def tag_chunk_vaccines(text: str) -> dict:
    """
    Build Pinecone metadata fields for vaccine mentions in a chunk.

    Returns:
        vaccines: list of vaccine ids mentioned
    """
    vaccine_ids = find_vaccines_in_text(text)
    return {"vaccines": vaccine_ids}
