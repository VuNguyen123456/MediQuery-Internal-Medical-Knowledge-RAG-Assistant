#!/usr/bin/env python3
"""
validate_screening.py — Smoke tests for vaccine detection and screening.

Run from repo root or services/rag/src:
  python services/rag/scripts/validate_screening.py

Requires Pinecone + env vars for full screening tests (steps 2–3).
Step 1 (registry detection) runs offline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add RAG src to path
_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from vaccines.registry import find_vaccines_in_text  # noqa: E402


def _ok(msg: str) -> None:
    print(f"  OK: {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    sys.exit(1)


def test_vaccine_detection() -> None:
    print("\n[1] Vaccine alias detection (offline)")
    found = find_vaccines_in_text("flu shot for adults")
    if "influenza" not in found:
        _fail(f"Expected 'influenza' in {found}")
    _ok(f"flu shot -> {found}")

    found2 = find_vaccines_in_text("MMR and varicella schedule for children")
    for expected in ("mmr", "varicella"):
        if expected not in found2:
            _fail(f"Expected '{expected}' in {found2}")
    _ok(f"MMR/varicella -> {found2}")


def test_vaccine_screening() -> None:
    print("\n[2] Vaccine precaution screening (requires Pinecone)")
    try:
        from vaccines.precautions import screen_vaccines
    except ImportError as exc:
        print(f"  SKIP: {exc}")
        return

    try:
        result = screen_vaccines(["influenza"], "pregnancy")
    except Exception as exc:
        _fail(f"screen_vaccines raised: {exc}")

    if not result.get("checks"):
        _fail("No checks returned")
    tier = result["checks"][0].get("tier")
    if tier not in ("red", "yellow", "green", "grey"):
        _fail(f"Invalid tier: {tier}")
    if not result.get("disclaimer"):
        _fail("Missing disclaimer")

    _ok(f"influenza x pregnancy -> tier={tier}, citations={len(result['checks'][0].get('citations', []))}")
    print(json.dumps(result, indent=2)[:1200] + "...")


def test_drug_screening_with_condition() -> None:
    print("\n[3] Drug interaction screening with condition (requires Pinecone)")
    try:
        from drugs.interactions import screen_drugs
    except ImportError as exc:
        print(f"  SKIP: {exc}")
        return

    try:
        result = screen_drugs(["metformin", "lisinopril"], condition="pregnancy")
    except Exception as exc:
        _fail(f"screen_drugs raised: {exc}")

    if len(result.get("pairs", [])) < 1:
        _fail("No pairs returned")
    note = result.get("condition_note", "")
    if "pregnancy" not in note.lower():
        _fail(f"condition_note missing pregnancy context: {note!r}")

    _ok(f"metformin x lisinopril with pregnancy -> {len(result['pairs'])} pair(s)")
    print(json.dumps(
        {
            "pairs": [{"label": p["label"], "tier": p["tier"]} for p in result["pairs"]],
            "condition_note": result.get("condition_note"),
        },
        indent=2,
    ))


def main() -> None:
    print("MediQuery screening validation")
    test_vaccine_detection()
    test_vaccine_screening()
    test_drug_screening_with_condition()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
