import sys
from pathlib import Path

try:
    from .utils.pdf_utils import pdf_to_json
except ImportError:  # pragma: no cover - supports direct script execution
    from utils.pdf_utils import pdf_to_json


def process_pdf(input_file_name: str, output_file_name: str | None = None) -> dict:
    """Extract text from a PDF file and save it as JSON."""
    input_path = Path(input_file_name)
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()

    output_path = Path(output_file_name) if output_file_name else input_path.with_suffix(".json")
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()

    return pdf_to_json(input_path, output_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python data_processing.py <input_pdf_filename>")

    result = process_pdf(sys.argv[1])
    print(f"Processed: {result['source']}")
    print(f"Pages extracted: {len(result['pages'])}")
