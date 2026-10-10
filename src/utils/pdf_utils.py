import json
from pathlib import Path
from typing import Any

from pypdf import PdfReader


def pdf_to_json(pdf_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    """Extract text from a PDF and return the data as JSON-friendly Python dicts."""
    source_path = Path(pdf_path)
    reader = PdfReader(str(source_path))

    pages: list[dict[str, Any]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({
            "page": page_number,
            "text": text.strip(),
        })

    payload = {
        "source": str(source_path),
        "pages": pages,
    }

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return payload
