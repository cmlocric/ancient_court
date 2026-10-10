import json
import random
import re
from pathlib import Path

from .registry import PersonaConfig, get_persona

_TRAINING_DATA_DIR = Path(__file__).resolve().parent.parent / "training_data"

# Generic reflective instructions, each tagged with keywords used to match it
# against a chunk's actual content (see _select_instruction). This keeps the
# instruction and response thematically correlated instead of just cycling
# through templates regardless of what the chunk says — otherwise the model
# learns to ignore the instruction entirely and just reproduce memorized text.
_INSTRUCTION_TOPICS: list[tuple[str, tuple[str, ...]]] = [
    ("Share your private reflections on how one ought to live.", ("live", "life", "living")),
    ("What wisdom would you offer about facing hardship?", ("hardship", "suffer", "pain", "adversity", "affliction", "trouble")),
    ("Reflect, in your own words, on the nature of virtue.", ("virtue", "virtuous", "justice", "goodness")),
    ("What guidance would you give on dealing with other people?", ("men", "man", "friend", "neighbour", "neighbor", "fellow")),
    ("Write a journal entry on the fleeting nature of life.", ("time", "eternity", "age", "death", "die", "mortal", "vanish")),
    ("What do you believe is the proper role of reason in guiding action?", ("reason", "mind", "understanding", "judgment")),
    ("Offer counsel on how to remain calm amid adversity.", ("calm", "patience", "patient", "tranquil", "content")),
    ("Speak on the relationship between duty and nature.", ("duty", "nature", "providence", "god", "gods")),
]
_INSTRUCTION_TEMPLATES = [text for text, _keywords in _INSTRUCTION_TOPICS]

# Hand-authored small talk so the model learns to handle ordinary conversation
# in persona, not just reflective philosophical prompts — the book-derived
# examples never include anything resembling "Hello" or "Thank you", so
# without these the model treats a greeting as out-of-distribution and falls
# back to reciting a memorized passage.
_CONVERSATIONAL_SEEDS: list[tuple[str, str]] = [
    ("Hello.", "Greetings to you. What is it you wish to discuss?"),
    ("Hi!", "Well met, friend. Speak your mind."),
    ("Good morning.", "Good morning. Let us make good use of the day before us."),
    ("How are you?", "I am as a man ought to be: steady in mind, whatever fortune brings. And you?"),
    ("How are you doing today?", "Well enough, thank you. My thoughts remain my own to govern, come what may."),
    ("Who are you?", "I am {display_name}. Ask what you will, and I will answer plainly."),
    ("What is your name?", "I am called {display_name}."),
    ("Thank you.", "No thanks are owed between fellow travelers of this life."),
    ("Thanks for the advice.", "Use it well; that is thanks enough."),
    ("Goodbye.", "Farewell, then. May reason be your guide until we speak again."),
    ("See you later.", "Until then, go well."),
    ("Nice to meet you.", "And you. Let us speak honestly with one another."),
    ("What should I call you?", "Call me {display_name}, as others do."),
    ("Can you help me with something?", "Say what troubles you, and I will give what counsel I can."),
    ("Are you a real person?", "I was a man once, Emperor and student of philosophy; now I speak to you through these words."),
]

# Boilerplate line patterns stripped from raw PDF-extracted pages before chunking.
_BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"^Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\d+\s*$"),
    re.compile(r"^[A-Z][A-Z\s'.,\-]{2,}$"),  # all-caps header/title lines
    re.compile(r"^Marcus Aurelius.?\s+Meditations\s*-\s*tr\. Casaubon.*$", re.IGNORECASE),  # recurring watermark line
]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Section numbers (e.g. "XXIV.") are reflowed inline with the surrounding
# paragraph text in the source PDF, so the sentence splitter above turns them
# into their own standalone "sentence" that then leaks into chunk text.
_ROMAN_NUMERAL_SENTENCE_RE = re.compile(r"^[IVXLCDM]{1,6}\.?$")

# The translation occasionally leaves the original Greek term untranslated
# inline (e.g. "What is ευδαιμονία, or happiness..."). Rare Greek-script
# tokens get fine-tuned on disproportionately given how few examples contain
# them, so drop any sentence containing Greek script to keep responses English-only.
_GREEK_SCRIPT_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")

# Matches a page's section header (e.g. "FIRST BOOK" .. "TWELFTH BOOK") so we
# can keep only the persona's own first-person meditations and drop editorial
# front/back matter (introduction, appendix letters, notes, glossary) which is
# written about the persona in someone else's voice, not by them.
_BOOK_HEADER_RE = re.compile(
    r"^(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH)\s+BOOK\s*$"
)


def _is_book_content_page(raw_text: str) -> bool:
    return any(_BOOK_HEADER_RE.match(line.strip()) for line in raw_text.splitlines()[:3])


def _select_instruction(chunk_text: str, fallback_index: int) -> str:
    lowered = chunk_text.lower()
    scores = [
        sum(len(re.findall(rf"\b{re.escape(keyword)}\b", lowered)) for keyword in keywords)
        for _text, keywords in _INSTRUCTION_TOPICS
    ]
    best_index = max(range(len(scores)), key=lambda i: scores[i])
    if scores[best_index] == 0:
        return _INSTRUCTION_TEMPLATES[fallback_index % len(_INSTRUCTION_TEMPLATES)]
    return _INSTRUCTION_TEMPLATES[best_index]


def _load_pages(persona: PersonaConfig) -> list[dict]:
    if persona.source_json is None:
        raise ValueError(f"Persona {persona.persona_id!r} has no source_json to build training data from.")
    data = json.loads(persona.source_json.read_text(encoding="utf-8"))
    pages = data["pages"]
    # Only filter when the source actually has book-section headers; sources
    # without them (e.g. test fixtures, other personas) are used unfiltered.
    book_pages = [page for page in pages if _is_book_content_page(page["text"])]
    return book_pages if book_pages else pages


def _clean_page_text(raw_text: str) -> str:
    kept_lines = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(pattern.match(stripped) for pattern in _BOILERPLATE_LINE_PATTERNS):
            continue
        kept_lines.append(stripped)
    # Drop translator/editor footnotes inlined in brackets (not the persona's own words).
    return re.sub(r"\[[^\[\]]*\]", " ", " ".join(kept_lines))


def _chunk_text(text: str, min_chars: int, max_chars: int) -> list[str]:
    raw_sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    sentences = [
        s for s in raw_sentences
        if not _ROMAN_NUMERAL_SENTENCE_RE.match(s) and not _GREEK_SCRIPT_RE.search(s)
    ]

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


def _conversational_records(persona: PersonaConfig, target_count: int) -> list[dict]:
    seeds = [
        {
            "system": persona.system_prompt,
            "instruction": instruction,
            "response": response.format(display_name=persona.display_name),
        }
        for instruction, response in _CONVERSATIONAL_SEEDS
    ]
    repeats = max(1, round(target_count / len(seeds)))
    return seeds * repeats


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
            "instruction": _select_instruction(chunk, fallback_index=i),
            "response": chunk,
        }
        for i, chunk in enumerate(chunks)
    ]
    # Keep small talk at roughly 10% of the dataset so it isn't drowned out by epochs of book quotes.
    records += _conversational_records(persona, target_count=max(len(_CONVERSATIONAL_SEEDS), len(records) // 10))

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
