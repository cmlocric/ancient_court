import json
import random
import re
from pathlib import Path

from .registry import PersonaConfig, get_persona

_TRAINING_DATA_DIR = Path(__file__).resolve().parent.parent / "training_data"

# Generic reflective instructions; intentionally persona-agnostic so the same
# pool works for any future persona added to the registry.
_INSTRUCTION_TEMPLATES = [
    "Share your private reflections on how one ought to live.",
    "What wisdom would you offer about facing hardship?",
    "Reflect, in your own words, on the nature of virtue.",
    "What guidance would you give on dealing with other people?",
    "Write a journal entry on the fleeting nature of life.",
    "What do you believe is the proper role of reason in guiding action?",
    "Offer counsel on how to remain calm amid adversity.",
    "Speak on the relationship between duty and nature.",
]

# Boilerplate line patterns stripped from raw PDF-extracted pages before chunking.
_BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"^Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\d+\s*$"),
    re.compile(r"^[A-Z][A-Z\s'.,\-]{2,}$"),  # all-caps header/title lines
]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _load_pages(persona: PersonaConfig) -> list[dict]:
    data = json.loads(persona.source_json.read_text(encoding="utf-8"))
    return data["pages"]


def _clean_page_text(raw_text: str) -> str:
    kept_lines = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(pattern.match(stripped) for pattern in _BOILERPLATE_LINE_PATTERNS):
            continue
        kept_lines.append(stripped)
    return " ".join(kept_lines)


def _chunk_text(text: str, min_chars: int, max_chars: int) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
        if len(current) >= min_chars and len(current) <= max_chars:
            chunks.append(current)
            current = ""

    if current and len(current) >= min_chars:
        chunks.append(current)

    return chunks


def build_dataset(
    persona_id: str,
    val_split: float = 0.1,
    output_dir: Path | None = None,
    min_chars: int = 200,
    max_chars: int = 600,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Build train/val JSONL instruction datasets for a persona from its source JSON."""
    persona = get_persona(persona_id)
    pages = _load_pages(persona)

    cleaned = " ".join(_clean_page_text(page["text"]) for page in pages)
    chunks = _chunk_text(cleaned, min_chars=min_chars, max_chars=max_chars)

    records = [
        {
            "system": persona.system_prompt,
            "instruction": _INSTRUCTION_TEMPLATES[i % len(_INSTRUCTION_TEMPLATES)],
            "response": chunk,
        }
        for i, chunk in enumerate(chunks)
    ]

    random.Random(seed).shuffle(records)
    split_index = int(len(records) * (1 - val_split))
    train_records, val_records = records[:split_index], records[split_index:]

    target_dir = Path(output_dir) if output_dir is not None else _TRAINING_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    train_path = target_dir / f"{persona_id}_train.jsonl"
    val_path = target_dir / f"{persona_id}_val.jsonl"

    for path, data in ((train_path, train_records), (val_path, val_records)):
        with path.open("w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return train_path, val_path


if __name__ == "__main__":
    import sys

    persona_arg = sys.argv[1] if len(sys.argv) > 1 else "marcus_aurelius"
    train_file, val_file = build_dataset(persona_arg)
    print(f"Train: {train_file}")
    print(f"Val:   {val_file}")
