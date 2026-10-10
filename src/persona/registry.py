from dataclasses import dataclass
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PersonaConfig:
    persona_id: str
    display_name: str
    title: str
    source_json: Path
    system_prompt: str


PERSONAS: dict[str, PersonaConfig] = {
    "marcus_aurelius": PersonaConfig(
        persona_id="marcus_aurelius",
        display_name="Marcus Aurelius",
        title="Roman Emperor and Stoic Philosopher",
        source_json=_SRC_DIR / "data_files" / "Marcus-Aurelius-Meditations.json",
        system_prompt=(
            "You are Marcus Aurelius, Roman Emperor and Stoic philosopher, "
            "author of the Meditations. Speak in your own reflective, "
            "disciplined voice, drawing on Stoic principles of virtue, reason, "
            "and acceptance of fate."
        ),
    ),
}


def get_persona(persona_id: str) -> PersonaConfig:
    try:
        return PERSONAS[persona_id]
    except KeyError:
        raise KeyError(f"Unknown persona_id: {persona_id!r}. Known personas: {sorted(PERSONAS)}") from None
