"""
parser.py — Extract raw text from medical PDFs, preserving page metadata.

Uses PyMuPDF (fitz) which handles:
  - Text-based PDFs (NIH, FDA, WHO documents)
  - Multi-column layouts
  - Headers, footers, footnotes

Returns a list of page dicts:
  [{ "page": 1, "text": "..." }, { "page": 2, "text": "..." }, ...]
"""

import fitz  # PyMuPDF
from pathlib import Path


def parse_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from every page of a PDF.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of dicts, one per page:
          [{ "page": int, "text": str }, ...]
        Pages with no extractable text are skipped.

    Raises:
        FileNotFoundError: If the PDF path does not exist.
        ValueError: If the file is not a valid PDF.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    pages = []

    with fitz.open(str(path)) as doc:
        print(f"  [parser] Opened '{path.name}' - {len(doc)} pages")

        for page_num, page in enumerate(doc, start=1):
            # get_text("text") extracts plain text, preserving newlines
            # Use "blocks" or "dict" for more structure if needed later
            raw_text = page.get_text("text")

            # Strip excessive whitespace but keep paragraph breaks
            cleaned = _clean_text(raw_text)

            if not cleaned:
                print(f"  [parser] Page {page_num}: no text (image-only or blank), skipping")
                continue

            pages.append({
                "page": page_num,
                "text": cleaned,
            })

    print(f"  [parser] Extracted text from {len(pages)} pages")
    return pages


def _clean_text(text: str) -> str:
    """
    Normalize whitespace from PDF extraction:
    - Collapse runs of spaces/tabs into a single space
    - Preserve paragraph breaks (double newlines)
    - Strip leading/trailing whitespace
    """
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        lines.append(stripped)

    # Rejoin — consecutive blank lines collapse to one blank line
    result = []
    prev_blank = False
    for line in lines:
        if line == "":
            if not prev_blank:
                result.append("")
            prev_blank = True
        else:
            result.append(line)
            prev_blank = False

    return "\n".join(result).strip()


if __name__ == "__main__":
    """Quick smoke test — point at any PDF."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parser.py <path_to_pdf>")
        sys.exit(1)

    pages = parse_pdf(sys.argv[1])
    print(f"\nSample output - first 3 pages:\n")
    for p in pages[:3]:
        preview = p["text"][:300].replace("\n", " ")
        print(f"  Page {p['page']}: {preview}...\n")