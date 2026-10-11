from src.web import app as web_app
from src.web.conversation import get_transcript


def _client(monkeypatch):
    monkeypatch.setattr(web_app, "generate", lambda pid, transcript, ckpt, present_persona_ids=(): f"{pid} reply")
    monkeypatch.setattr(
        web_app, "generate_debate_reply", lambda pid, transcript, ckpt, present_persona_ids=(): f"{pid} rebuttal"
    )
    return web_app.create_app().test_client()


def test_message_then_one_advisor_at_a_time(monkeypatch):
    client = _client(monkeypatch)
    sid = "route-session"
    council = ["marcus_aurelius", "machiavelli"]

    assert client.post("/chat/message", json={"session_id": sid, "message": "Hi"}).status_code == 200
    first = client.post("/chat/respond", json={"session_id": sid, "persona_id": "marcus_aurelius", "active_personas": council})
    second = client.post(
        "/chat/respond",
        json={"session_id": sid, "persona_id": "machiavelli", "active_personas": council, "mode": "debate"},
    )

    assert first.get_json()["turn"] == {"speaker": "marcus_aurelius", "text": "marcus_aurelius reply"}
    assert second.get_json()["turn"] == {"speaker": "machiavelli", "text": "machiavelli rebuttal"}
    assert [t["speaker"] for t in get_transcript(sid)] == ["user", "marcus_aurelius", "machiavelli"]


def test_respond_rejects_debate_with_one_advisor_and_unknown_persona(monkeypatch):
    client = _client(monkeypatch)

    debate = client.post(
        "/chat/respond",
        json={"session_id": "s", "persona_id": "machiavelli", "active_personas": ["machiavelli"], "mode": "debate"},
    )
    unknown = client.post("/chat/respond", json={"session_id": "s", "persona_id": "nobody", "active_personas": []})

    assert debate.status_code == 400
    assert unknown.status_code == 400
