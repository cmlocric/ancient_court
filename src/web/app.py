import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.persona.infer import generate, generate_greeting
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

    @app.post("/chat/send")
    def chat_send():
        payload = request.get_json(force=True) or {}
        session_id = payload.get("session_id") or str(uuid.uuid4())
        message = (payload.get("message") or "").strip()
        active_personas = payload.get("active_personas") or []

        if not message:
            return jsonify({"error": "message is required"}), 400
        unknown = [pid for pid in active_personas if pid not in PERSONAS]
        if unknown:
            return jsonify({"error": f"unknown persona_id(s): {unknown}"}), 400

        append_turn(session_id, "user", message)
        new_turns = [{"speaker": "user", "text": message}]

        for persona_id in active_personas:
            transcript = get_transcript(session_id)
            checkpoint_dir = _resolve_checkpoint_dir(persona_id)
            reply = generate(persona_id, transcript, checkpoint_dir)
            append_turn(session_id, persona_id, reply)
            new_turns.append({"speaker": persona_id, "text": reply})

        return jsonify({"session_id": session_id, "turns": new_turns})

    @app.post("/persona/activate")
    def persona_activate():
        payload = request.get_json(force=True) or {}
        session_id = payload.get("session_id") or str(uuid.uuid4())
        persona_id = payload.get("persona_id")

        if persona_id not in PERSONAS:
            return jsonify({"error": f"unknown persona_id: {persona_id}"}), 400

        # Loads (and caches) the model on first activation, then has the persona
        # greet the court instead of waiting for the first user question.
        transcript = get_transcript(session_id)
        checkpoint_dir = _resolve_checkpoint_dir(persona_id)
        greeting = generate_greeting(persona_id, transcript, checkpoint_dir)
        append_turn(session_id, persona_id, greeting)

        return jsonify({"session_id": session_id, "turn": {"speaker": persona_id, "text": greeting}})

    return app


if __name__ == "__main__":
    # Threaded so a persona activation (model load) doesn't block other requests.
    create_app().run(debug=True, threaded=True)
