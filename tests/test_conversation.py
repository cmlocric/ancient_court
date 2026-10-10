from src.persona.infer import build_prompt_for_persona
from src.persona.registry import PersonaConfig
from src.web.conversation import append_turn, get_transcript


def _fake_persona() -> PersonaConfig:
    return PersonaConfig(
        persona_id="fake_persona",
        display_name="Fake Persona",
        title="Test Persona",
        source_json="unused.json",
        system_prompt="You are a fake persona used only for testing.",
    )


def test_append_and_get_transcript_preserves_order():
    session_id = "session-1"
    append_turn(session_id, "user", "Hello there")
    append_turn(session_id, "fake_persona", "Greetings, traveler.")

    transcript = get_transcript(session_id)

    assert transcript == [
        {"speaker": "user", "text": "Hello there"},
        {"speaker": "fake_persona", "text": "Greetings, traveler."},
    ]


def test_get_transcript_is_isolated_per_session():
    append_turn("session-a", "user", "Message A")
    append_turn("session-b", "user", "Message B")

    assert get_transcript("session-a") == [{"speaker": "user", "text": "Message A"}]
    assert get_transcript("session-b") == [{"speaker": "user", "text": "Message B"}]


def test_build_prompt_for_persona_includes_system_prompt_and_turns_in_order():
    persona = _fake_persona()
    transcript = [
        {"speaker": "user", "text": "What is virtue?"},
        {"speaker": "other_persona", "text": "Virtue is its own reward."},
    ]

    prompt = build_prompt_for_persona(persona, transcript)

    system_index = prompt.index(persona.system_prompt)
    user_index = prompt.index("What is virtue?")
    other_index = prompt.index("Virtue is its own reward.")
    final_marker_index = prompt.rindex(f"<|{persona.persona_id}|>")

    assert system_index < user_index < other_index < final_marker_index
