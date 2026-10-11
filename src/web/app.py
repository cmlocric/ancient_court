import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.persona.infer import generate, generate_debate_reply, generate_greeting
from src.persona.registry import PERSONAS
from src.web.conversation import append_turn, get_transcript

_CHECKPOINTS_ROOT = Path(__file__).resolve().parent.parent.parent / "checkpoints"


def _resolve_checkpoint_dir(persona_id: str) -> Path | None:
    # Fine-tuned adapters currently hurt answer quality, so default to the base model + persona prompt.
    if os.environ.get("PERSONA_USE_ADAPTER") != "1":
        return None
    # Allows pointing a persona at a custom checkpoint (e.g. a smoke-test run)
    # without changing code: MARCUS_AURELIUS_CHECKPOINT_DIR=checkpoints/marcus_aurelius_smoke
    override = os.environ.get(f"{persona_id.upper()}_CHECKPOINT_DIR")
    return Path(override) if override else _CHECKPOINTS_ROOT / persona_id


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        personas = [
            {"persona_id": p.persona_id, "display_name": p.display_name, "title": p.title}
            for p in PERSONAS.values()
        ]
        return render_template("chat.html", personas=personas)

    def _validate_personas(active_personas):
        unknown = [pid for pid in active_personas if pid not in PERSONAS]
        if unknown:
            return jsonify({"error": f"unknown persona_id(s): {unknown}"}), 400
        return None

    @app.post("/chat/message")
    def chat_message():
        payload = request.get_json(force=True) or {}
        session_id = payload.get("session_id") or str(uuid.uuid4())
        message = (payload.get("message") or "").strip()

        if not message:
            return jsonify({"error": "message is required"}), 400

        append_turn(session_id, "user", message)
        return jsonify({"session_id": session_id, "turn": {"speaker": "user", "text": message}})

    # One advisor per request, so the page can show and speak each reply while the next one generates.
    @app.post("/chat/respond")
    def chat_respond():
        payload = request.get_json(force=True) or {}
        session_id = payload.get("session_id") or str(uuid.uuid4())
        persona_id = payload.get("persona_id")
        active_personas = payload.get("active_personas") or []
        mode = payload.get("mode", "reply")

        error = _validate_personas(active_personas + [persona_id])
        if error:
            return error
        if mode not in ("reply", "debate"):
            return jsonify({"error": f"unknown mode: {mode}"}), 400
        if mode == "debate" and len(active_personas) < 2:
            return jsonify({"error": "a debate needs at least two active advisors"}), 400

        transcript = get_transcript(session_id)
        checkpoint_dir = _resolve_checkpoint_dir(persona_id)
        generator = generate_debate_reply if mode == "debate" else generate
        reply = generator(persona_id, transcript, checkpoint_dir, present_persona_ids=active_personas)
        append_turn(session_id, persona_id, reply)

        return jsonify({"session_id": session_id, "turn": {"speaker": persona_id, "text": reply}})

    @app.post("/persona/activate")
    def persona_activate():
        payload = request.get_json(force=True) or {}
        session_id = payload.get("session_id") or str(uuid.uuid4())
        persona_id = payload.get("persona_id")
        active_personas = [pid for pid in (payload.get("active_personas") or []) if pid in PERSONAS]

        if persona_id not in PERSONAS:
            return jsonify({"error": f"unknown persona_id: {persona_id}"}), 400

        # Loads (and caches) the model on first activation, then has the persona
        # greet the court instead of waiting for the first user question.
        transcript = get_transcript(session_id)
        checkpoint_dir = _resolve_checkpoint_dir(persona_id)
        greeting = generate_greeting(
            persona_id, transcript, checkpoint_dir, present_persona_ids=active_personas + [persona_id]
        )
        append_turn(session_id, persona_id, greeting)

        return jsonify({"session_id": session_id, "turn": {"speaker": persona_id, "text": greeting}})

    return app


if __name__ == "__main__":
    # Threaded so a persona activation (model load) doesn't block other requests.
    create_app().run(debug=True, threaded=True)
