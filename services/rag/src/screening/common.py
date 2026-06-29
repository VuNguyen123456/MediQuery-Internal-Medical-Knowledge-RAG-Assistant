"""
common.py — Shared chunk merge and citation helpers for screening modules.
"""

from __future__ import annotations

from generation.prompt import _format_excerpt


def chunk_key(chunk: dict) -> str:
    return f"{chunk.get('source')}::{chunk.get('page')}::{chunk.get('chunk_index')}"


def merge_chunks(*groups: list[dict]) -> list[dict]:
    seen: set[str] = set()
    merged: list[dict] = []
    for group in groups:
        for chunk in group:
            key = chunk_key(chunk)
            if key in seen:
                continue
            seen.add(key)
            merged.append(chunk)
    merged.sort(key=lambda c: c.get("score", 0), reverse=True)
    return merged


def chunks_to_citations(chunks: list[dict], limit: int = 4) -> list[dict]:
    citations = []
    seen: set[str] = set()
    for chunk in chunks[:limit]:
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", 0)
        key = f"{source}::{page}"
        if key in seen:
            continue
        seen.add(key)
        citations.append({
            "source": source,
            "page": int(page) if isinstance(page, float) and page == int(page) else page,
            "excerpt": _format_excerpt(chunk.get("text", "")),
            "score": chunk.get("score", 0),
        })
    return citations
