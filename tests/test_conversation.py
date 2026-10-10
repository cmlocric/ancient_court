from src.persona.infer import build_messages_for_persona
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


def test_build_messages_for_persona_maps_roles_in_order():
    persona = _fake_persona()
    transcript = [
        {"speaker": "user", "text": "What is virtue?"},
        {"speaker": "fake_persona", "text": "Virtue is its own reward."},
        {"speaker": "other_persona", "text": "I disagree."},
    ]

    messages = build_messages_for_persona(persona, transcript)

    assert messages[0]["role"] == "system"
    assert persona.system_prompt in messages[0]["content"]
    assert messages[1:] == [
        {"role": "user", "content": "What is virtue?"},
        {"role": "assistant", "content": "Virtue is its own reward."},
        {"role": "user", "content": "other_persona: I disagree."},
    ]


def test_build_messages_for_persona_names_other_present_advisors():
    persona = _fake_persona()

    messages = build_messages_for_persona(persona, [], present_persona_ids=["fake_persona", "marcus_aurelius"])

    assert "Also present in this council: Marcus Aurelius." in messages[0]["content"]
    assert "You are Fake Persona:" in messages[0]["content"]
    assert "Fake Persona." not in messages[0]["content"]
