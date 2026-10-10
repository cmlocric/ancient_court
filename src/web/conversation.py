import threading

from src.persona.infer import build_messages_for_persona

# In-memory conversation store: session_id -> list of turns.
# No database; state resets on server restart (acceptable for this POC).
_SESSIONS: dict[str, list[dict]] = {}
_LOCK = threading.Lock()


def append_turn(session_id: str, speaker: str, text: str) -> None:
    with _LOCK:
        _SESSIONS.setdefault(session_id, []).append({"speaker": speaker, "text": text})


def get_transcript(session_id: str) -> list[dict]:
    with _LOCK:
        return list(_SESSIONS.get(session_id, []))


__all__ = ["append_turn", "get_transcript", "build_messages_for_persona"]
