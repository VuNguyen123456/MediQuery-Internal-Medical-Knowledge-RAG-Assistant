"""
catalog.py — Document root resolution, sections config, and PDF listing.

PDFs live under documents/<section-id>/*.pdf
Section definitions live in documents/sections.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent
_SECTIONS_FILE = "sections.json"
_SECTION_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def resolve_documents_root() -> Path:
    """
    Resolve the documents folder for local dev and Docker.

    Docker:  /app/documents   (WORKDIR /app, src at /app/src)
    Local:   <project_root>/documents
    """
    docker_docs = _SRC_DIR.parent / "documents"
    if docker_docs.is_dir() and (_SRC_DIR.parent / "src").is_dir():
        return docker_docs

    local_docs = _SRC_DIR.parent.parent.parent / "documents"
    return local_docs


def _sections_config_path(root: Path | None = None) -> Path:
    return (root or resolve_documents_root()) / _SECTIONS_FILE


def default_sections() -> list[dict]:
    return [
        {"id": "drug-labels", "label": "Drug Labels", "order": 1},
        {"id": "guidelines", "label": "Clinical Guidelines", "order": 2},
        {"id": "vaccines", "label": "Vaccines & Immunization", "order": 3},
        {"id": "toolkits", "label": "Clinical Toolkits", "order": 4},
    ]


def load_sections(root: Path | None = None) -> list[dict]:
    """Load section definitions; create defaults if missing."""
    docs_root = root or resolve_documents_root()
    config_path = _sections_config_path(docs_root)

    if config_path.is_file():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            sections = raw.get("sections", [])
            if sections:
                return sorted(sections, key=lambda s: s.get("order", 99))
        except (json.JSONDecodeError, OSError):
            pass

    return default_sections()


def ensure_sections_file(root: Path | None = None) -> None:
    """Write default sections.json and section subfolders if needed."""
    docs_root = root or resolve_documents_root()
    docs_root.mkdir(parents=True, exist_ok=True)
    config_path = _sections_config_path(docs_root)

    if not config_path.is_file():
        payload = {"sections": default_sections()}
        config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for section in load_sections(docs_root):
        section_id = section.get("id", "")
        if section_id and _SECTION_ID_RE.match(section_id):
            section_path = docs_root / section_id
            if section_path.exists() and not section_path.is_dir():
                continue
            section_path.mkdir(parents=True, exist_ok=True)


def is_valid_section_id(section_id: str) -> bool:
    return bool(section_id and _SECTION_ID_RE.match(section_id))


def resolve_section_id(section_id: str | None, root: Path | None = None) -> str | None:
    """Return section id if valid and defined in sections.json."""
    if not section_id or not is_valid_section_id(section_id.strip()):
        return None
    sid = section_id.strip()
    known = {s["id"] for s in load_sections(root)}
    return sid if sid in known else None


def relative_pdf_path(pdf_path: Path, root: Path | None = None) -> str:
    """POSIX-style path relative to documents root (e.g. drug-labels/Metformin.pdf)."""
    docs_root = root or resolve_documents_root()
    rel = pdf_path.relative_to(docs_root).as_posix()
    return rel


def section_id_for_pdf(pdf_path: Path, root: Path | None = None) -> str:
    docs_root = root or resolve_documents_root()
    rel = pdf_path.relative_to(docs_root)
    parts = rel.parts
    if len(parts) >= 2:
        candidate = parts[0]
        known = {s["id"] for s in load_sections(docs_root)}
        if candidate in known:
            return candidate
    return "general"


def iter_pdfs(root: Path | None = None) -> list[Path]:
    """All PDFs under documents root (section subfolders), sorted."""
    docs_root = root or resolve_documents_root()
    if not docs_root.is_dir():
        return []
    return sorted(docs_root.glob("**/*.pdf"))


def count_pdfs(root: Path | None = None) -> int:
    return len(iter_pdfs(root))


def pdf_to_document_entry(pdf_path: Path, root: Path | None = None) -> dict:
    docs_root = root or resolve_documents_root()
    rel = relative_pdf_path(pdf_path, docs_root)
    return {
        "path": rel,
        "name": pdf_path.name,
        "section_id": section_id_for_pdf(pdf_path, docs_root),
        "size_kb": round(pdf_path.stat().st_size / 1024, 1),
    }


def build_documents_payload(root: Path | None = None) -> dict:
    """
    Group PDFs by section for the sidebar.

    Returns:
        sections: [{ id, label, order, documents: [...] }]
        documents: flat list (backwards compatible)
    """
    docs_root = root or resolve_documents_root()
    ensure_sections_file(docs_root)

    section_defs = load_sections(docs_root)
    section_map: dict[str, dict] = {
        s["id"]: {
            "id": s["id"],
            "label": s.get("label", s["id"]),
            "order": s.get("order", 99),
            "documents": [],
        }
        for s in section_defs
    }
    section_map.setdefault(
        "general",
        {"id": "general", "label": "General", "order": 0, "documents": []},
    )

    flat: list[dict] = []
    for pdf_path in iter_pdfs(docs_root):
        entry = pdf_to_document_entry(pdf_path, docs_root)
        flat.append(entry)
        bucket = section_map.get(entry["section_id"], section_map["general"])
        bucket["documents"].append(entry)

    sections_out = sorted(
        [s for s in section_map.values() if s["documents"]],
        key=lambda s: s.get("order", 99),
    )

    return {"sections": sections_out, "documents": flat}


def find_pdf_by_path(relative_path: str, root: Path | None = None) -> Path | None:
    """Resolve a relative document path safely (no path traversal)."""
    docs_root = root or resolve_documents_root()
    clean = relative_path.replace("\\", "/").strip().lstrip("/")
    if not clean.lower().endswith(".pdf"):
        return None
    if ".." in clean.split("/"):
        return None

    candidate = (docs_root / clean).resolve()
    try:
        candidate.relative_to(docs_root.resolve())
    except ValueError:
        return None

    return candidate if candidate.is_file() else None


def destination_for_upload(
    filename: str,
    section_id: str | None,
    root: Path | None = None,
) -> tuple[Path, str] | tuple[None, str]:
    """
    Return (absolute dest path, relative source path) for an upload.

    On error returns (None, error_message).
    """
    docs_root = root or resolve_documents_root()
    ensure_sections_file(docs_root)

    basename = Path(filename).name
    if not basename.lower().endswith(".pdf"):
        return None, "Only PDF files are supported"

    sid = resolve_section_id(section_id, docs_root)
    if not sid:
        return None, "Invalid or missing section. Choose a section for the upload."

    section_dir = docs_root / sid
    section_dir.mkdir(parents=True, exist_ok=True)
    dest = section_dir / basename
    rel = relative_pdf_path(dest, docs_root)
    return dest, rel
