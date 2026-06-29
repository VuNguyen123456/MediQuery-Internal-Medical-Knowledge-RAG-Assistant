"""
tagging.py — Tag chunks with drug metadata during ingestion.
"""

from __future__ import annotations

from drugs.registry import find_drugs_in_text


def tag_chunk_drugs(text: str) -> dict:
    """
    Build Pinecone metadata fields for drug mentions in a chunk.

    Returns:
        drugs: list of drug ids mentioned
        drug_pair: sorted "a|b" key when 2+ drugs co-mentioned (else omitted)
    """
    drug_ids = find_drugs_in_text(text)
    meta: dict = {"drugs": drug_ids}

    if len(drug_ids) >= 2:
        pair_key = "|".join(sorted(drug_ids))
        meta["drug_pair"] = pair_key

    return meta
